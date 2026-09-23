import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qaitu.metrics_models import (Attribution, MetricSource, MetricValue, MetricsRun,
                                  PeriodComparison, Recommendation, StructuralEvent)
from qaitu.metrics_service import _validate, metrics_enabled, run_metrics
from qaitu.metrics_store import load_knowledge, persist_run, restore_decisions, save_decision, update_knowledge


def fixture(synthetic=True, day="2022-12-23", scope="ДККМ"):
    value = MetricValue("mv1", "report_delay_days", 3, scope, "2021-01-01", "2021-01-31", "month",
                        MetricSource("SYNTH.xlsx" if synthetic else "real.xlsx", "Задержка 3 дня", "Свод", "B2"), synthetic)
    event = StructuralEvent("W1", "func_weakened", [scope], day, [{"clause_id": "5.5.3", "quote": "отчёты"}],
                            mechanism="Периодичность отчётности", mechanism_confirmed=True, is_synthetic=synthetic)
    comparison = PeriodComparison("c1", value.metric, scope, event.id, {"value": 3, "n_points": 3},
                                  {"value": 11, "n_points": 3}, 8, 266.67, method="trend_adjusted", metric_ids=[value.id], is_synthetic=synthetic)
    attribution = Attribution("a1", event.id, comparison.id, "После изменения задержка выросла", event.mechanism,
                              "medium", ["прямой механизм"], ["Найм"], {"metrics": [value.id]}, is_synthetic=synthetic)
    return MetricsRun("metrics_test", synthetic, [value], [event], [comparison], [attribution])


class StoreTests(unittest.TestCase):
    def test_reupload_is_one_case_and_namespaces_are_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture()
            self.assertEqual(update_knowledge(run, root)[0].n_cases, 1)
            run.run_id = "another_run"
            run.attributions[0].id = "new_attribution_id"
            run.comparisons[0].delta_abs = 9
            self.assertEqual(update_knowledge(run, root)[0].n_cases, 1)
            self.assertEqual(load_knowledge(False, root), [])
            self.assertEqual(update_knowledge(fixture(False), root)[0].n_cases, 1)
            self.assertEqual(load_knowledge(True, root)[0].n_cases, 1)
            self.assertEqual(Path(root, "data/kb.sqlite").stat().st_mode & 0o777, 0o600)

    def test_quantitative_pattern_requires_three_independent_cases(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(3):
                knowledge = update_knowledge(fixture(day=f"202{index}-12-23"), root)
                self.assertEqual(knowledge[0].n_cases, index + 1)
                self.assertEqual(knowledge[0].observed_effect["quantitative_allowed"], index == 2)
            run = fixture(day="2022-12-23")
            run.attributions[0].confidence = "insufficient"
            knowledge = update_knowledge(run, root)
            self.assertEqual(knowledge[0].n_cases, 2)
            self.assertFalse(knowledge[0].observed_effect["quantitative_allowed"])

    def test_decisions_and_all_stages_persist(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture()
            run.recommendations = [Recommendation("r1", "Уточнить регламент", "restore_function", {"attributions": ["a1"]}, {"metric": "report_delay_days"}, "medium", [])]
            save_decision(run, "r1", "accept", root)
            folder = Path(root, "runs", run.run_id, "metrics")
            self.assertEqual(len(list(folder.glob("M*.json"))), 9)
            record = json.loads((folder / "M9_report.json").read_text())
            self.assertEqual(record["recommendations"][0]["decision"], "accept")
            run.recommendations[0].decision = "discuss"
            restore_decisions(run, root)
            self.assertEqual(run.recommendations[0].decision, "accept")
            self.assertEqual((folder / "M9_report.json").stat().st_mode & 0o777, 0o600)

    def test_unsafe_run_path_and_mixed_data_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run = fixture()
            run.run_id = "../escape"
            with self.assertRaises(ValueError):
                persist_run(run, root)
        run = fixture()
        run.events[0].is_synthetic = False
        with self.assertRaisesRegex(ValueError, "смешивать"):
            _validate(run.values, run.events, [])

    def test_provenance_required_and_flag_honored(self):
        run = fixture()
        run.values[0].source.cell = None
        with self.assertRaisesRegex(ValueError, "источник"):
            _validate(run.values, run.events, [])
        with patch.dict("os.environ", {"METRICS_MODULE": "off"}):
            self.assertFalse(metrics_enabled())
            with self.assertRaisesRegex(ValueError, "отключён"):
                run_metrics([], [])
