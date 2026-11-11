from ..contracts import (
    identifier,
    integer,
    canonical,
    decode,
    digest,
    fail,
    fields as object_fields,
    pair as make_pair,
    sequence as bounded_sequence,
)
from .freeze import DatasetFreeze
from .methods import (
    METHODS,
    RawScoreCalibrator,
    classify,
    evaluate,
    precision_recall_at,
    choose_thresholds,
)


class Calibration:
    """匹配分数校准实验编排。

    流程：人工标签台账 -> 冻结记录版本与特征 -> 连通分量切分 -> 训练集拟合
    两种校准方法 -> 验证集比较并按误合并风险上限选阈值 -> 留出集只评估一次。
    任何自动阈值只在无标签冲突、样本充足且风险约束可满足时才可自动使用。
    """

    def __init__(self, store, ledger, evidence, sources, review):
        self.store = store
        self.ledger = ledger
        self.evidence = evidence
        self.sources = sources
        self.review = review
        self.freezer = DatasetFreeze(store, ledger, evidence, sources)

    # -- 冻结与切分 --------------------------------------------------------
    def freeze_dataset(
        self,
        freeze_id,
        now,
        holdout_fraction=0.2,
        validation_fraction=0.2,
        min_bucket_pairs=10,
        max_component_fraction=0.5,
        large_component="block",
    ):
        return self.freezer.freeze(
            freeze_id,
            now,
            holdout_fraction,
            validation_fraction,
            min_bucket_pairs,
            max_component_fraction,
            large_component,
        )

    def get_freeze(self, freeze_id):
        return self.freezer.get(freeze_id)

    def _load_freeze(self, freeze_id):
        frozen = self.freezer.get(freeze_id)
        current = {
            "data": self.store.revision("data"),
            "dictionary": self.store.revision("dictionary"),
            "graph": self.store.revision("graph"),
        }
        if current != frozen["revisions"]:
            fail(
                "stale",
                "store revisions moved after freeze; freeze a new dataset",
                frozen_revisions=frozen["revisions"],
                current_revisions=current,
            )
        return frozen

    # -- 训练 + 验证 -------------------------------------------------------
    def run_validation(
        self,
        experiment_id,
        freeze_id,
        now,
        risk_cap=0.05,
        bins=10,
        grid=20,
        min_auto=1,
    ):
        identifier(experiment_id)
        integer(now)
        frozen = self._load_freeze(freeze_id)
        if self.store.one(
            "SELECT id FROM calibration_experiments WHERE freeze_id=? AND role='validation'",
            (freeze_id,),
        ):
            fail("conflict", "validation experiment already exists for this freeze")
        if self.store.one(
            "SELECT id FROM calibration_experiments WHERE freeze_id=? AND role='holdout'",
            (freeze_id,),
        ):
            fail("sequence", "holdout already evaluated; validation cannot be rerun")

        train = DatasetFreeze.binary_points(frozen, "train")
        validation = DatasetFreeze.binary_points(frozen, "validation")
        blockers = self._blockers(frozen, train, validation)

        methods_report = {}
        calibrators = {}
        selected = next(iter(METHODS))
        if not blockers:
            for name, cls in METHODS.items():
                fitted = cls.fit(train)
                calibrators[name] = fitted
                metrics = evaluate(validation, fitted, bins)
                methods_report[name] = {
                    "parameters": fitted.params(),
                    "validation": metrics,
                }
            baseline = evaluate(validation, RawScoreCalibrator(), bins)
            methods_report[RawScoreCalibrator.name] = {
                "parameters": RawScoreCalibrator().params(),
                "validation": baseline,
            }

            # 按验证集 Brier 选择主方法（平局时等渗在前的固定排序兜底）
            selected = min(
                METHODS,
                key=lambda name: (
                    methods_report[name]["validation"]["brier"],
                    0 if name == "isotonic_pava" else 1,
                ),
            )

        policy = {"usable": False, "reasons": list(blockers)}
        if not blockers:
            choice = choose_thresholds(
                validation, calibrators[selected], risk_cap, grid, min_auto
            )
            if choice["usable"]:
                pr = precision_recall_at(
                    validation,
                    calibrators[selected],
                    choice["low"],
                    choice["high"],
                )
                policy = {
                    "usable": True,
                    "reasons": [],
                    "method": selected,
                    "low": choice["low"],
                    "high": choice["high"],
                    "risk_cap": risk_cap,
                    "threshold_search": {
                        "high": choice["high_search"],
                        "low": choice["low_search"],
                    },
                    "validation_precision_recall": pr,
                }
            else:
                policy = {
                    "usable": False,
                    "reasons": [
                        "no threshold satisfies the mis-merge risk cap on validation data"
                    ],
                    "method": selected,
                    "threshold_search": {
                        "high": choice["high_search"],
                        "low": choice["low_search"],
                    },
                }
            for name, calibrator in calibrators.items():
                methods_report[name]["validation"]["precision_recall"] = (
                    precision_recall_at(
                        validation,
                        calibrator,
                        policy.get("low", 0.0),
                        policy.get("high", 1.0),
                    )
                    if policy["usable"]
                    else None
                )

        payload = {
            "id": experiment_id,
            "freeze_id": freeze_id,
            "freeze_digest": frozen["digest"],
            "role": "validation",
            "created_at": now,
            "settings": {
                "risk_cap": risk_cap,
                "bins": bins,
                "grid": grid,
                "min_auto": min_auto,
            },
            "bucket_counts": {
                "train": len(train),
                "validation": len(validation),
            },
            "selected_method": selected,
            "methods": methods_report,
            "policy": policy,
            "unknown_pair_ids": DatasetFreeze.unknown_pair_ids(
                frozen, "validation"
            ),
        }
        checksum = digest(payload)
        self._save_experiment(experiment_id, freeze_id, "validation", payload, now)
        return {**payload, "digest": checksum}

    # -- 留出集：只报告一次 -----------------------------------------------
    def evaluate_holdout(self, experiment_id, freeze_id, now, bins=10):
        identifier(experiment_id)
        integer(now)
        frozen = self._load_freeze(freeze_id)
        if self.store.one(
            "SELECT id FROM calibration_experiments WHERE freeze_id=? AND role='holdout'",
            (freeze_id,),
        ):
            fail(
                "sequence",
                "holdout set may be reported only once for a frozen dataset",
            )
        validation_row = self.store.one(
            "SELECT payload FROM calibration_experiments WHERE freeze_id=? AND role='validation'",
            (freeze_id,),
        )
        if validation_row is None:
            fail("sequence", "run validation before holdout evaluation")
        validation = decode(validation_row["payload"])
        holdout = DatasetFreeze.binary_points(frozen, "holdout")

        blockers = []
        if not holdout:
            blockers.append("holdout bucket has no usable binary labels")
        summary = frozen["bucket_summary"]["holdout"]
        if not summary["meets_minimum"]:
            blockers.append(
                "holdout bucket below min_bucket_pairs="
                + str(frozen["parameters"]["min_bucket_pairs"])
            )

        methods_report = {}
        if holdout and validation["methods"]:
            for name in METHODS:
                cls = METHODS[name]
                params = validation["methods"][name]["parameters"]
                calibrator = self._restore(cls, params)
                methods_report[name] = {
                    "holdout": evaluate(holdout, calibrator, bins)
                }
            methods_report[RawScoreCalibrator.name] = {
                "holdout": evaluate(holdout, RawScoreCalibrator(), bins)
            }
        else:
            for name in [*METHODS, RawScoreCalibrator.name]:
                methods_report[name] = {"holdout": None}

        policy = validation["policy"]
        auto_usable = policy.get("usable", False) and not blockers
        holdout_policy = None
        if policy.get("usable") and holdout:
            selected = policy["method"]
            cls = METHODS[selected]
            calibrator = self._restore(
                cls, validation["methods"][selected]["parameters"]
            )
            holdout_policy = precision_recall_at(
                holdout, calibrator, policy["low"], policy["high"]
            )
        payload = {
            "id": experiment_id,
            "freeze_id": freeze_id,
            "freeze_digest": frozen["digest"],
            "role": "holdout",
            "selected_method": validation["selected_method"],
            "created_at": now,
            "bucket_counts": {"holdout": len(holdout)},
            "methods": methods_report,
            "validated_policy": policy,
            "holdout_policy_check": holdout_policy,
            "auto_threshold_usable": auto_usable,
            "blockers": blockers,
            "unknown_pair_ids": DatasetFreeze.unknown_pair_ids(frozen, "holdout"),
            "note": "single final report; thresholds were chosen on validation only",
        }
        checksum = digest(payload)
        self._save_experiment(experiment_id, freeze_id, "holdout", payload, now)
        return {**payload, "digest": checksum}

    # -- 建议进入审查队列（只建议，不改图，批次幂等） ----------------------
    def suggest_batch(
        self,
        batch_id,
        freeze_id,
        validation_experiment_id,
        now,
        pairs,
        include_manual=True,
    ):
        identifier(batch_id)
        integer(now)
        bounded_sequence(pairs, 1, 10000)
        if not isinstance(include_manual, bool):
            fail("input", "include_manual must be boolean")
        normalized_pairs = []
        for item in pairs:
            object_fields(item, {"left", "right"}, {"left", "right"})
            left, right = make_pair(item["left"], item["right"])
            normalized_pairs.append({"left": left, "right": right})
        pairs = normalized_pairs
        frozen = self._load_freeze(freeze_id)
        validation = self.get_experiment(validation_experiment_id)
        if validation["role"] != "validation":
            fail("input", "thresholds must come from a validation experiment")
        request_digest = digest(
            {
                "freeze_id": freeze_id,
                "experiment_id": validation_experiment_id,
                "pairs": pairs,
                "include_manual": include_manual,
            }
        )
        with self.store.atomic():
            existing = self.store.one(
                "SELECT id,payload FROM calibration_suggestions WHERE batch_digest=?",
                (request_digest,),
            )
            if existing:
                return decode(existing["payload"])
            if self.store.one(
                "SELECT id FROM calibration_suggestions WHERE id=?", (batch_id,)
            ):
                fail("conflict", "suggestion batch id reused with different payload")

        policy = validation["policy"]
        selected = policy.get("method")
        calibrator = None
        if policy.get("usable"):
            cls = METHODS[selected]
            calibrator = self._restore(
                cls, validation["methods"][selected]["parameters"]
            )

        suggestions = []
        created_tasks = []
        for index, pair in enumerate(pairs):
            evidence = self.evidence.compare(pair["left"], pair["right"])
            if policy.get("usable"):
                probability = calibrator.predict(evidence["score"])
                advice = classify(probability, policy["low"], policy["high"])
            else:
                probability = None
                advice = "manual_review"
            if advice == "manual_review" and not include_manual:
                continue
            task_id = "cal-" + request_digest[:16] + "-" + str(index)
            task = self.review.enqueue(task_id, pair["left"], pair["right"])
            created_tasks.append(task_id)
            suggestions.append(
                {
                    "task_id": task_id,
                    "left": pair["left"],
                    "right": pair["right"],
                    "advice": advice,
                    "calibrated_probability": (
                        round(probability, 6) if probability is not None else None
                    ),
                    "rule_score": evidence["score"],
                    "review_state": task["state"],
                }
            )

        payload = {
            "id": batch_id,
            "batch_digest": request_digest,
            "freeze_id": freeze_id,
            "validation_experiment_id": validation_experiment_id,
            "created_at": now,
            "auto_threshold_usable": policy.get("usable", False),
            "suggestions": suggestions,
            "created_review_tasks": created_tasks,
            "note": "advisory only; identity graph is never modified by suggestions",
        }
        checksum = digest(payload)
        with self.store.atomic() as con:
            con.execute(
                "INSERT INTO calibration_suggestions VALUES(?,?,?,?,?,?)",
                (
                    batch_id,
                    request_digest,
                    validation_experiment_id,
                    canonical(payload),
                    checksum,
                    now,
                ),
            )
            self.store.audit(
                "calibration_suggestions_enqueued",
                batch_id,
                {
                    "digest": request_digest,
                    "tasks": len(created_tasks),
                    "auto_threshold_usable": policy.get("usable", False),
                },
            )
        return {**payload, "digest": checksum}

    def list_suggestions(self):
        result = []
        for row in self.store.rows(
            "SELECT * FROM calibration_suggestions ORDER BY created_at,id"
        ):
            payload = decode(row["payload"])
            if digest(payload) != row["digest"]:
                fail("integrity", "suggestion payload changed")
            result.append({**payload, "digest": row["digest"]})
        return result

    # -- 读取 --------------------------------------------------------------
    def get_experiment(self, experiment_id):
        row = self.store.one(
            "SELECT * FROM calibration_experiments WHERE id=?", (experiment_id,)
        )
        if row is None:
            fail("missing", "experiment not found")
        payload = decode(row["payload"])
        if digest(payload) != row["digest"]:
            fail("integrity", "experiment payload changed")
        return {**payload, "digest": row["digest"]}

    def list_experiments(self, freeze_id=None):
        if freeze_id is None:
            rows = self.store.rows(
                "SELECT * FROM calibration_experiments ORDER BY created_at,id"
            )
        else:
            rows = self.store.rows(
                "SELECT * FROM calibration_experiments WHERE freeze_id=? ORDER BY created_at,id",
                (freeze_id,),
            )
        result = []
        for row in rows:
            payload = decode(row["payload"])
            if digest(payload) != row["digest"]:
                fail("integrity", "experiment payload changed")
            result.append({**payload, "digest": row["digest"]})
        return result

    # -- 内部 --------------------------------------------------------------
    def _blockers(self, frozen, train, validation):
        blockers = []
        if frozen["conflicting_pair_ids"]:
            blockers.append(
                "hard label conflicts present ("
                + str(len(frozen["conflicting_pair_ids"]))
                + " pairs); human reconciliation required"
            )
        if len(train) < 2:
            blockers.append("training bucket has fewer than two binary pairs")
        if not validation:
            blockers.append("validation bucket has no binary pairs")
        elif not frozen["bucket_summary"]["validation"]["meets_minimum"]:
            blockers.append(
                "validation bucket below min_bucket_pairs="
                + str(frozen["parameters"]["min_bucket_pairs"])
            )
        return blockers

    def _restore(self, cls, params):
        if cls.__name__ == "PlattCalibrator":
            return cls(params["a"], params["b"])
        if cls.__name__ == "IsotonicCalibrator":
            blocks = [
                (block["score"], block["probability"], block["weight"])
                for block in params["blocks"]
            ]
            return cls(blocks)
        fail("state", "unknown calibrator parameters")

    def _save_experiment(self, experiment_id, freeze_id, role, payload, now):
        checksum = digest(payload)
        with self.store.atomic() as con:
            con.execute(
                "INSERT INTO calibration_experiments"
                "(id,freeze_id,role,method,payload,digest,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    freeze_id,
                    role,
                    payload.get("selected_method"),
                    canonical(payload),
                    checksum,
                    now,
                ),
            )
            self.store.audit(
                "calibration_experiment_recorded",
                experiment_id,
                {"freeze_id": freeze_id, "role": role, "digest": checksum},
            )
