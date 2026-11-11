from ..contracts import identifier, integer, canonical, decode, digest, fail


class DatasetFreeze:
    """冻结标签实验的记录版本与特征，并按连通分量切分数据集。

    切分图的节点是记录，边是全部带有效标签的记录对（含无法判断与冲突对，
    保证任何记录都不会跨桶出现）。连通分量整体分配，分量内的候选边绝不漏
    到另一个桶。切分完全确定，不使用随机数。
    """

    def __init__(self, store, ledger, evidence, sources):
        self.store = store
        self.ledger = ledger
        self.evidence = evidence
        self.sources = sources

    def _components(self, pair_views):
        parent = {}

        def root(key):
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def join(a, b):
            ra, rb = root(a), root(b)
            if ra != rb:
                # 字典序较小者作根，保证结果确定
                if ra > rb:
                    ra, rb = rb, ra
                parent[rb] = ra

        for view in pair_views:
            root(view["left"])
            root(view["right"])
            join(view["left"], view["right"])
        grouped = {}
        for key in sorted(parent):
            grouped.setdefault(root(key), set()).add(key)
        return [sorted(members) for members in grouped.values()]

    def freeze(
        self,
        freeze_id,
        now,
        holdout_fraction=0.2,
        validation_fraction=0.2,
        min_bucket_pairs=10,
        max_component_fraction=0.5,
        large_component="block",
    ):
        identifier(freeze_id)
        integer(now)
        integer(min_bucket_pairs, 1, 100000)
        if large_component not in {"block", "degrade"}:
            fail("input", "large_component must be block or degrade")
        for value, name in [
            (holdout_fraction, "holdout_fraction"),
            (validation_fraction, "validation_fraction"),
            (max_component_fraction, "max_component_fraction"),
        ]:
            if not isinstance(value, (int, float)) or not 0 < value < 1:
                fail("input", name + " must be a fraction strictly between 0 and 1")
        if holdout_fraction + validation_fraction >= 1:
            fail("input", "holdout and validation fractions must leave a training set")

        if self.store.one(
            "SELECT id FROM calibration_freeze WHERE id=?", (freeze_id,)
        ):
            return self.get(freeze_id)

        self.sources.require_index()
        views = self.ledger.pair_view()
        if not views:
            fail("input", "no active labels to freeze")
        hard_conflicts = [
            view["pair_id"] for view in views if view["hard_conflict"]
        ]
        if hard_conflicts:
            fail(
                "conflict",
                "hard label conflicts must be reconciled or withdrawn before freezing",
                conflicting_pairs=hard_conflicts,
            )

        frozen_pairs = []
        for view in views:
            evidence = self.evidence.compare(view["left"], view["right"])
            frozen_pairs.append(
                {
                    "pair_id": view["pair_id"],
                    "left": view["left"],
                    "right": view["right"],
                    "labels": view["labels"],
                    "status": view["status"],
                    "hard_conflict": view["hard_conflict"],
                    "sources": view["sources"],
                    "entries": view["entries"],
                    "evidence": evidence,
                    "evidence_changed_since_label": self._label_digest_mismatch(
                        view, evidence
                    ),
                }
            )

        components = self._components(views)
        pair_by_id = {item["pair_id"]: item for item in frozen_pairs}
        view_by_id = {view["pair_id"]: view for view in views}
        # 每个分量包含的边数（全部标签边）
        comp_edges = []
        for index, members in enumerate(components):
            member_set = set(members)
            edges = [
                pid
                for pid, view in view_by_id.items()
                if view["left"] in member_set and view["right"] in member_set
            ]
            comp_edges.append(edges)
        total_edges = sum(len(edges) for edges in comp_edges)

        # 二分类可用边：状态 agreed 且标签为 same/different
        def binary_edges(edge_ids):
            return [
                pid
                for pid in edge_ids
                if pair_by_id[pid]["status"] == "agreed"
                and pair_by_id[pid]["labels"][0] in ("same", "different")
            ]

        ordering = sorted(
            range(len(components)),
            key=lambda i: (-len(comp_edges[i]), components[i][0]),
        )
        targets = {
            "holdout": holdout_fraction,
            "validation": validation_fraction,
            "train": 1 - holdout_fraction - validation_fraction,
        }
        bucket_of_component = {}
        bucket_pairs = {"train": 0, "validation": 0, "holdout": 0}
        blocked = []
        degraded = []
        for i in ordering:
            edge_count = len(comp_edges[i])
            if total_edges and edge_count / total_edges > max_component_fraction:
                if large_component == "block":
                    blocked.append(
                        {
                            "component_root": components[i][0],
                            "records": len(components[i]),
                            "edges": edge_count,
                            "fraction": _round6(edge_count / total_edges),
                        }
                    )
                    continue
                degraded.append(
                    {
                        "component_root": components[i][0],
                        "records": len(components[i]),
                        "edges": edge_count,
                        "fraction": _round6(edge_count / total_edges),
                        "placement": "train",
                    }
                )
                chosen = "train"
            else:
                # 确定性贪心：放进目标缺口最大的桶，桶优先级固定打破平局
                chosen = max(
                    ("holdout", "validation", "train"),
                    key=lambda bucket: (
                        targets[bucket] * total_edges - bucket_pairs[bucket],
                        -{"holdout": 2, "validation": 1, "train": 0}[bucket],
                    ),
                )
            bucket_of_component[i] = chosen
            bucket_pairs[chosen] += edge_count

        if blocked:
            fail(
                "blocked",
                "connected component too large to split without record leakage",
                components=blocked,
                policy="reduce component size or rerun with large_component=degrade",
            )

        record_bucket = {}
        for i, bucket in bucket_of_component.items():
            for key in components[i]:
                record_bucket[key] = bucket
        for item in frozen_pairs:
            item["bucket"] = record_bucket[item["left"]]
            if record_bucket[item["right"]] != item["bucket"]:  # 防御性检查
                fail("integrity", "component split leaked across buckets")

        buckets = {"train": [], "validation": [], "holdout": []}
        for item in frozen_pairs:
            buckets[item["bucket"]].append(item["pair_id"])

        summary = {}
        for bucket in ("train", "validation", "holdout"):
            binary = binary_edges(buckets[bucket])
            same = sum(1 for pid in binary if pair_by_id[pid]["labels"][0] == "same")
            summary[bucket] = {
                "all_labeled_pairs": len(buckets[bucket]),
                "binary_pairs": len(binary),
                "same": same,
                "different": len(binary) - same,
                "unknown_pairs": sum(
                    1
                    for pid in buckets[bucket]
                    if pair_by_id[pid]["labels"] == ["unknown"]
                ),
                "conflicting_pairs": sum(
                    1 for pid in buckets[bucket] if pair_by_id[pid]["hard_conflict"]
                ),
                "meets_minimum": len(binary) >= min_bucket_pairs,
            }

        payload = {
            "id": freeze_id,
            "created_at": now,
            "revisions": {
                "data": self.store.revision("data"),
                "dictionary": self.store.revision("dictionary"),
                "graph": self.store.revision("graph"),
            },
            "algorithm": "explainable-v1",
            "parameters": {
                "holdout_fraction": holdout_fraction,
                "validation_fraction": validation_fraction,
                "min_bucket_pairs": min_bucket_pairs,
                "max_component_fraction": max_component_fraction,
                "large_component": large_component,
            },
            "pairs": frozen_pairs,
            "components": [
                {
                    "root": components[i][0],
                    "bucket": bucket_of_component.get(i),
                    "records": components[i],
                    "edges": comp_edges[i],
                }
                for i in range(len(components))
            ],
            "buckets": buckets,
            "bucket_summary": summary,
            "degraded_components": degraded,
            "conflicting_pair_ids": [
                item["pair_id"] for item in frozen_pairs if item["hard_conflict"]
            ],
        }
        checksum = digest(payload)
        with self.store.atomic() as con:
            con.execute(
                "INSERT INTO calibration_freeze VALUES(?,?,?,?)",
                (freeze_id, canonical(payload), checksum, now),
            )
            self.store.audit(
                "calibration_frozen",
                freeze_id,
                {
                    "digest": checksum,
                    "components": len(components),
                    "pairs": len(frozen_pairs),
                    "degraded": len(degraded),
                },
            )
        return self.get(freeze_id)

    def _label_digest_mismatch(self, view, current_evidence):
        stale = []
        for entry in view["entries"]:
            full = self.ledger.get(entry["id"])
            if full["evidence_digest"] != current_evidence["digest"]:
                stale.append(entry["id"])
        return stale

    def get(self, freeze_id):
        row = self.store.one(
            "SELECT * FROM calibration_freeze WHERE id=?", (freeze_id,)
        )
        if row is None:
            fail("missing", "frozen dataset not found")
        payload = decode(row["payload"])
        if digest(payload) != row["digest"]:
            fail("integrity", "frozen dataset changed")
        return {**payload, "digest": row["digest"]}

    @staticmethod
    def _pair_index(frozen):
        return {pair["pair_id"]: pair for pair in frozen["pairs"]}

    @staticmethod
    def binary_points(frozen, bucket):
        index = DatasetFreeze._pair_index(frozen)
        points = []
        for pid in frozen["buckets"][bucket]:
            item = index[pid]
            if item["status"] != "agreed":
                continue
            label = item["labels"][0]
            if label not in ("same", "different"):
                continue
            points.append(
                {
                    "pair_id": pid,
                    "left": item["left"],
                    "right": item["right"],
                    "score": item["evidence"]["score"],
                    "y": 1 if label == "same" else 0,
                }
            )
        return points

    @staticmethod
    def unknown_pair_ids(frozen, bucket=None):
        index = DatasetFreeze._pair_index(frozen)
        result = []
        buckets = [bucket] if bucket else ("train", "validation", "holdout")
        for name in buckets:
            for pid in frozen["buckets"][name]:
                if index[pid]["labels"] == ["unknown"]:
                    result.append(pid)
        return result


def _round6(value):
    return round(float(value), 6)
