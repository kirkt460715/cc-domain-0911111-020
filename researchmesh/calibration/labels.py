from ..contracts import (
    identifier,
    bounded_text,
    integer,
    canonical,
    decode,
    digest,
    pair,
    fail,
)

LABELS = {"same", "different", "unknown"}


class LabelLedger:
    """人工确认标签台账。

    标签只记录人工来源的判断，绝不从既有合并决定推导真值。同一记录对可保留
    多条标签历史：重复提交幂等返回，矛盾标签同时可见并标记冲突，撤回保留
    可追溯记录而不是物理删除。
    """

    def __init__(self, store, evidence):
        self.store = store
        self.evidence = evidence

    def record(self, label_id, left, right, label, source, now, note="", supersedes=None):
        identifier(label_id)
        identifier(source)
        left, right = pair(left, right)
        bounded_text(note, 2000, True)
        if label not in LABELS:
            fail("input", "label must be same, different or unknown")
        integer(now)
        pair_id = left + "|" + right
        if supersedes is not None:
            identifier(supersedes)
        with self.store.atomic() as con:
            old = self.store.one(
                "SELECT * FROM calibration_labels WHERE id=?", (label_id,)
            )
            if old:
                current = decode(old["payload"])
                if (
                    current["left"],
                    current["right"],
                    current["label"],
                    current["source"],
                    current["note"],
                    current["supersedes"],
                ) != (left, right, label, source, note, supersedes):
                    fail("conflict", "label id already used with another payload")
                return self.get(label_id)
            if supersedes is not None:
                prior = self.store.one(
                    "SELECT * FROM calibration_labels WHERE id=?", (supersedes,)
                )
                if prior is None:
                    fail("missing", "superseded label not found")
                if decode(prior["payload"])["pair_id"] != pair_id:
                    fail("input", "superseded label belongs to another pair")
            # 冻结时刻的证据快照，保证标签针对的规则分数与特征可复盘
            evidence = self.evidence.compare(left, right)
            payload = {
                "id": label_id,
                "pair_id": pair_id,
                "left": left,
                "right": right,
                "label": label,
                "source": source,
                "note": note,
                "supersedes": supersedes,
                "recorded_at": now,
                "evidence_digest": evidence["digest"],
                "rule_score": evidence["score"],
                "conflicts": evidence["conflicts"],
            }
            con.execute(
                "INSERT INTO calibration_labels VALUES(?,?,?,?,?,?,?, 'active',?,NULL,NULL,?,?,?)",
                (
                    label_id,
                    pair_id,
                    left,
                    right,
                    label,
                    source,
                    note,
                    now,
                    supersedes,
                    canonical(payload),
                    digest(payload),
                ),
            )
            self.store.audit(
                "calibration_label_recorded",
                label_id,
                {
                    "pair_id": pair_id,
                    "label": label,
                    "source": source,
                    "supersedes": supersedes,
                },
            )
        return self.get(label_id)

    def withdraw(self, label_id, now, reason):
        identifier(label_id)
        reason = bounded_text(reason, 2000)
        integer(now)
        with self.store.atomic() as con:
            row = self.store.one(
                "SELECT * FROM calibration_labels WHERE id=?", (label_id,)
            )
            if row is None:
                fail("missing", "label not found")
            if row["state"] != "active":
                fail("state", "label already withdrawn")
            con.execute(
                "UPDATE calibration_labels SET state='withdrawn',withdrawn_at=?,"
                "withdraw_reason=? WHERE id=?",
                (now, reason, label_id),
            )
            payload = decode(row["payload"])
            payload["state"] = "withdrawn"
            payload["withdrawn_at"] = now
            payload["withdraw_reason"] = reason
            con.execute(
                "UPDATE calibration_labels SET payload=?,digest=? WHERE id=?",
                (canonical(payload), digest(payload), label_id),
            )
            self.store.audit(
                "calibration_label_withdrawn",
                label_id,
                {"pair_id": row["pair_id"], "reason": reason},
            )
        return self.get(label_id)

    def get(self, label_id):
        row = self.store.one(
            "SELECT * FROM calibration_labels WHERE id=?", (label_id,)
        )
        if row is None:
            fail("missing", "label not found")
        payload = decode(row["payload"])
        if digest(payload) != row["digest"]:
            fail("integrity", "label payload changed")
        return payload

    def _active_rows(self):
        return self.store.rows(
            "SELECT * FROM calibration_labels WHERE state='active' ORDER BY id"
        )

    def pair_view(self):
        """按记录对汇总标签：来源、多数状态、冲突全部可见。"""
        grouped = {}
        for row in self._active_rows():
            grouped.setdefault(row["pair_id"], []).append(decode(row["payload"]))
        result = []
        for pair_id in sorted(grouped):
            entries = sorted(grouped[pair_id], key=lambda item: item["id"])
            labels = sorted({item["label"] for item in entries})
            hard_conflict = {"same", "different"} <= set(labels)
            if hard_conflict:
                status = "conflicting"
            elif len(labels) > 1:
                status = "disputed"  # 与 unknown 并存但无同/不同直接对立
            else:
                status = "agreed"
            result.append(
                {
                    "pair_id": pair_id,
                    "left": entries[0]["left"],
                    "right": entries[0]["right"],
                    "labels": labels,
                    "status": status,
                    "hard_conflict": hard_conflict,
                    "sources": sorted({item["source"] for item in entries}),
                    "entries": [
                        {
                            "id": item["id"],
                            "label": item["label"],
                            "source": item["source"],
                            "note": item["note"],
                            "recorded_at": item["recorded_at"],
                            "supersedes": item["supersedes"],
                        }
                        for item in entries
                    ],
                }
            )
        return result

    def history(self):
        rows = self.store.rows(
            "SELECT id,pair_id,label,source,state,recorded_at,withdrawn_at,withdraw_reason "
            "FROM calibration_labels ORDER BY recorded_at,id"
        )
        return [dict(row) for row in rows]

    def conflicts(self):
        return [view for view in self.pair_view() if view["hard_conflict"]]

    def stats(self):
        counts = {"same": 0, "different": 0, "unknown": 0, "withdrawn": 0}
        pairs = {"agreed": 0, "disputed": 0, "conflicting": 0}
        for row in self.store.rows("SELECT state,label FROM calibration_labels"):
            if row["state"] == "withdrawn":
                counts["withdrawn"] += 1
            else:
                counts[row["label"]] += 1
        for view in self.pair_view():
            pairs[view["status"]] += 1
        return {"label_counts": counts, "pair_counts": pairs}
