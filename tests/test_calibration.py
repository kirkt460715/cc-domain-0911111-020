import unittest
from researchmesh.service import ResearchMesh
from researchmesh.contracts import MeshError


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.m = ResearchMesh()
        self.addCleanup(self.m.close)
        self.m.sources.register("a", "source a")
        self.m.sources.register("b", "source b")

    def record(self, title, doi=None, inst="Lab", kind="dataset", year=2026):
        row = {"title": title, "institution": inst, "kind": kind, "year": year}
        if doi:
            row["identifiers"] = {"doi": [doi]}
        return row

    def seed(self, same_pairs=60, diff_pairs=40):
        a_rows, b_rows = [], []
        for i in range(same_pairs):
            a_rows.append(
                {"id": f"s{i}", "data": self.record(f"Sensor study {i}", f"10.1234/s{i}")}
            )
            b_rows.append(
                {"id": f"t{i}", "data": self.record(f"sensor study {i}", f"10.1234/s{i}")}
            )
        for j in range(diff_pairs):
            a_rows.append(
                {
                    "id": f"u{j}",
                    "data": self.record(
                        f"Quantum widget {j}",
                        f"10.1234/qw{j}",
                        kind="software",
                        inst="North Lab",
                    ),
                }
            )
            b_rows.append(
                {
                    "id": f"v{j}",
                    "data": self.record(
                        f"Farm manual {j}",
                        f"10.1234/fm{j}",
                        kind="process",
                        inst="South Lab",
                    ),
                }
            )
        self.m.sources.snapshot("a1", "a", 1, a_rows, 0)
        self.m.sources.snapshot("b1", "b", 1, b_rows, 0)
        now = 1000
        for i in range(same_pairs):
            self.m.labels.record(
                f"same-{i}", f"a:s{i}", f"b:t{i}", "same", "reviewer-1", now
            )
        for j in range(diff_pairs):
            self.m.labels.record(
                f"diff-{j}", f"a:u{j}", f"b:v{j}", "different", "reviewer-2", now
            )
        return now

    # -- 标签台账 ----------------------------------------------------------
    def test_label_ledger_tracks_source_conflict_and_withdrawal(self):
        self.seed(same_pairs=2, diff_pairs=2)
        self.m.labels.record(
            "extra", "a:s0", "b:t0", "different", "reviewer-9", 1001
        )
        conflicts = self.m.labels.conflicts()
        self.assertEqual([c["pair_id"] for c in conflicts], ["a:s0|b:t0"])
        view = next(c for c in self.m.labels.pair_view() if c["hard_conflict"])
        self.assertEqual(view["status"], "conflicting")
        self.assertEqual(sorted(view["sources"]), ["reviewer-1", "reviewer-9"])

        withdrawn = self.m.labels.withdraw("extra", 1002, "wrong pair")
        self.assertEqual(withdrawn["state"], "withdrawn")
        self.assertEqual(self.m.labels.conflicts(), [])
        history = self.m.labels.history()
        self.assertIn(
            ("extra", "withdrawn"), {(h["id"], h["state"]) for h in history}
        )

    def test_label_recording_is_idempotent_and_validated(self):
        self.seed(same_pairs=1, diff_pairs=1)
        first = self.m.labels.record(
            "dup", "a:s0", "b:t0", "same", "reviewer-1", 1000
        )
        second = self.m.labels.record(
            "dup", "a:s0", "b:t0", "same", "reviewer-1", 1000
        )
        self.assertEqual(first["id"], second["id"])
        with self.assertRaises(MeshError) as caught:
            self.m.labels.record(
                "dup", "a:s0", "b:t0", "different", "reviewer-1", 1000
            )
        self.assertEqual(caught.exception.code, "conflict")
        with self.assertRaises(MeshError):
            self.m.labels.record(
                "bad", "a:s0", "b:t0", "maybe", "reviewer-1", 1000
            )

    def test_existing_merge_decisions_are_not_used_as_truth(self):
        self.seed(same_pairs=2, diff_pairs=1)
        # 即使图中存在人工合并决定，标签台账仍为空时不得冻结
        preview = self.m.graph.preview("a:s0", "b:t0")
        self.m.graph.decide(
            "merge-1",
            "a:s0",
            "b:t0",
            "merge",
            preview["digest"],
            "reviewer",
            "manual",
        )
        fresh = ResearchMesh()
        self.addCleanup(fresh.close)
        with self.assertRaises(MeshError) as caught:
            fresh.sources.register("x", "x")
            fresh.calibration.freeze_dataset("f", 1)
        self.assertEqual(caught.exception.code, "input")

    # -- 冻结与切分 --------------------------------------------------------
    def test_freeze_splits_connected_components_without_leakage(self):
        self.seed()
        frozen = self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=5, holdout_fraction=0.2,
            validation_fraction=0.2,
        )
        record_buckets = {}
        for component in frozen["components"]:
            self.assertIsNotNone(component["bucket"])
            for key in component["records"]:
                self.assertNotIn(key, record_buckets)
                record_buckets[key] = component["bucket"]
        # 每条边两端记录必须在同一桶
        for pair in frozen["pairs"]:
            self.assertEqual(
                record_buckets[pair["left"]],
                record_buckets[pair["right"]],
                pair["pair_id"],
            )
            self.assertEqual(pair["bucket"], record_buckets[pair["left"]])
        self.assertEqual(frozen["revisions"]["data"], self.m.store.revision("data"))
        # 冻结内容带摘要且可重复读取
        again = self.m.calibration.get_freeze("f1")
        self.assertEqual(again["digest"], frozen["digest"])

    def test_hard_conflict_blocks_freeze_until_reconciled(self):
        self.seed(same_pairs=3, diff_pairs=2)
        self.m.labels.record("c", "a:s0", "b:t0", "different", "r", 1001)
        with self.assertRaises(MeshError) as caught:
            self.m.calibration.freeze_dataset("f", 1002, min_bucket_pairs=1)
        self.assertEqual(caught.exception.code, "conflict")
        self.m.labels.withdraw("c", 1003, "corrected")
        frozen = self.m.calibration.freeze_dataset("f", 1004, min_bucket_pairs=1)
        self.assertEqual(frozen["conflicting_pair_ids"], [])

    def test_large_component_blocks_or_degrades_explicitly(self):
        self.seed(same_pairs=12, diff_pairs=2)
        # 用共享记录串成一个大分量：s0 出现在多条边中
        for j in range(1, 8):
            self.m.labels.record(
                f"chain-{j}", f"a:s0", f"b:t{j}", "same", "reviewer-1", 1000
            )
        with self.assertRaises(MeshError) as caught:
            self.m.calibration.freeze_dataset(
                "f", 1002, min_bucket_pairs=1, max_component_fraction=0.3
            )
        self.assertEqual(caught.exception.code, "blocked")
        self.assertTrue(caught.exception.details["components"])
        frozen = self.m.calibration.freeze_dataset(
            "f",
            1003,
            min_bucket_pairs=1,
            max_component_fraction=0.3,
            large_component="degrade",
        )
        self.assertTrue(frozen["degraded_components"])
        self.assertEqual(frozen["degraded_components"][0]["placement"], "train")

    def test_freeze_rejects_when_store_revisions_move(self):
        self.seed(same_pairs=4, diff_pairs=4)
        self.m.calibration.freeze_dataset("f", 1002, min_bucket_pairs=1)
        self.m.dictionary.put("institution", "Lab", "Canonical Lab", 0, "policy")
        self.m.sources.reindex(1)
        with self.assertRaises(MeshError) as caught:
            self.m.calibration.run_validation("e", "f", 1003, min_auto=1)
        self.assertEqual(caught.exception.code, "stale")

    # -- 校准实验 ----------------------------------------------------------
    def test_validation_compares_methods_and_reports_metrics(self):
        self.seed()
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=5, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        # 每桶约 25 个二分类标签，零错误时 Wilson 上界可支持 0.05 风险上限
        report = self.m.calibration.run_validation(
            "e-val", "f1", 2001, bins=5, risk_cap=0.3
        )
        self.assertEqual(
            set(report["methods"]),
            {"platt_logistic", "isotonic_pava", "raw_score_baseline"},
        )
        for name in ("platt_logistic", "isotonic_pava"):
            validation = report["methods"][name]["validation"]
            self.assertIn("brier", validation)
            self.assertIn("reliability_bins", validation)
            self.assertEqual(
                sum(b["count"] for b in validation["reliability_bins"]),
                validation["n"],
            )
        self.assertIn(report["selected_method"], ("platt_logistic", "isotonic_pava"))
        self.assertTrue(report["policy"]["usable"])
        self.assertLessEqual(report["policy"]["low"], report["policy"]["high"])
        pr = report["policy"]["validation_precision_recall"]
        self.assertIsNotNone(pr["precision"])
        self.assertIsNotNone(pr["recall"])

    def test_holdout_is_reported_exactly_once(self):
        self.seed()
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=5, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        self.m.calibration.run_validation(
            "e-val", "f1", 2001, bins=5, risk_cap=0.3
        )
        holdout = self.m.calibration.evaluate_holdout("e-hold", "f1", 2002, bins=5)
        self.assertTrue(holdout["auto_threshold_usable"])
        self.assertIsNotNone(holdout["holdout_policy_check"])
        for name in ("platt_logistic", "isotonic_pava", "raw_score_baseline"):
            self.assertIn("brier", holdout["methods"][name]["holdout"])
        with self.assertRaises(MeshError) as caught:
            self.m.calibration.evaluate_holdout("e-hold-2", "f1", 2003)
        self.assertEqual(caught.exception.code, "sequence")
        with self.assertRaises(MeshError):
            self.m.calibration.run_validation("e-val-2", "f1", 2004)

    def test_insufficient_validation_data_yields_no_auto_threshold(self):
        self.seed(same_pairs=5, diff_pairs=3)
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=1, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        report = self.m.calibration.run_validation(
            "e-val", "f1", 2001, risk_cap=0.01, min_auto=1
        )
        # 样本太少时严格风险上限无法满足 -> 不给自动阈值
        self.assertFalse(report["policy"]["usable"])
        self.assertTrue(report["policy"]["reasons"])

    def test_validation_blocker_skips_fitting_and_still_records_report(self):
        self.seed(same_pairs=4, diff_pairs=3)
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=100, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        report = self.m.calibration.run_validation("e-val", "f1", 2001)
        self.assertFalse(report["policy"]["usable"])
        self.assertTrue(report["policy"]["reasons"])
        self.assertEqual(report["methods"], {})
        # 留出仍可出一次性报告，但不可自动使用
        holdout = self.m.calibration.evaluate_holdout("e-hold", "f1", 2002)
        self.assertFalse(holdout["auto_threshold_usable"])

    def test_holdout_below_minimum_reports_blocker_without_threshold(self):
        self.seed(same_pairs=6, diff_pairs=4)
        frozen = self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=50, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        self.assertFalse(frozen["bucket_summary"]["holdout"]["meets_minimum"])
        self.m.calibration.run_validation(
            "e-val", "f1", 2001, risk_cap=0.4, min_auto=1
        )
        holdout = self.m.calibration.evaluate_holdout("e-hold", "f1", 2002)
        self.assertFalse(holdout["auto_threshold_usable"])
        self.assertTrue(holdout["blockers"])

    # -- 建议入队 ----------------------------------------------------------
    def test_suggestions_are_advisory_idempotent_and_never_edit_graph(self):
        self.seed()
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=5, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        self.m.calibration.run_validation(
            "e-val", "f1", 2001, bins=5, risk_cap=0.3
        )
        pairs = [
            {"left": "a:s0", "right": "b:t0"},
            {"left": "a:u0", "right": "b:v0"},
        ]
        before = len(self.m.graph.clusters())
        batch = self.m.calibration.suggest_batch(
            "s1", "f1", "e-val", 2002, pairs
        )
        advices = {s["advice"] for s in batch["suggestions"]}
        self.assertEqual(advices, {"suggest_same", "suggest_different"})
        for suggestion in batch["suggestions"]:
            task = self.m.review.get(suggestion["task_id"])
            self.assertEqual(task["state"], "pending")
        # 身份图不变
        self.assertEqual(len(self.m.graph.clusters()), before)

        pending_before = len(self.m.review.pending(3000)["tasks"])
        repeated = self.m.calibration.suggest_batch(
            "s1-again", "f1", "e-val", 2003, pairs
        )
        pending_after = len(self.m.review.pending(3000)["tasks"])
        self.assertEqual(repeated["batch_digest"], batch["batch_digest"])
        self.assertEqual(repeated["id"], batch["id"])
        self.assertEqual(pending_before, pending_after)

    def test_unusable_policy_routes_everything_to_manual_review(self):
        self.seed(same_pairs=5, diff_pairs=3)
        self.m.calibration.freeze_dataset(
            "f1", 2000, min_bucket_pairs=1, holdout_fraction=0.25,
            validation_fraction=0.25,
        )
        self.m.calibration.run_validation(
            "e-val", "f1", 2001, risk_cap=0.01
        )
        batch = self.m.calibration.suggest_batch(
            "s1", "f1", "e-val", 2002,
            [{"left": "a:s0", "right": "b:t0"}],
        )
        self.assertFalse(batch["auto_threshold_usable"])
        self.assertEqual(batch["suggestions"][0]["advice"], "manual_review")
        self.assertIsNone(batch["suggestions"][0]["calibrated_probability"])

    def test_dispatch_exposes_calibration_actions(self):
        result = self.m.dispatch(
            {"action": "label.stats", "arguments": {}}
        )
        self.assertEqual(
            result["label_counts"],
            {"same": 0, "different": 0, "unknown": 0, "withdrawn": 0},
        )
        actions = {item["action"] for item in self.m.describe()}
        self.assertIn("calibration.run_validation", actions)
        self.assertIn("calibration.suggest_batch", actions)
        self.assertIn("label.record", actions)


if __name__ == "__main__":
    unittest.main()
