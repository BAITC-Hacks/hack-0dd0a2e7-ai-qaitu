from pathlib import Path
import re
import tempfile
import unittest

from scripts.gen_synthetic_reports import generate_reports
from qaitu.metrics_ingest import ingest_reports
from qaitu.metrics_models import StructuralEvent, Hypothesis
from qaitu.metrics_service import run_metrics, render_markdown


class DemoAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        manifest = generate_reports(Path(cls.directory.name) / "reports")
        result = ingest_reports([(Path(p).name, Path(p).read_bytes()) for p in manifest["files"]], is_synthetic=True)
        cls.metrics_run = run_metrics(result.values, [StructuralEvent(**e) for e in manifest["events"]],
                              hypotheses=[Hypothesis(**h) for h in manifest["hypotheses"]],
                              control_scopes=manifest["controls"], as_of="2023-12-31", storage_root=cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_no_causal_assertions_in_full_demo_report(self):
        report = render_markdown(self.metrics_run)
        self.assertIsNone(re.search(r"привело\s+к|вызвало|благодаря", report, re.I))

    def test_seeded_effects_sources_and_hiring_confound(self):
        by_event = {a.event_id: a for a in self.metrics_run.attributions}
        for event in ("W1", "S1"):
            self.assertIn(event, by_event)
            self.assertIn(by_event[event].confidence, {"medium", "high"})
        self.assertIn("найм", str(by_event["S1"].alternative_explanations).lower())
        sources = {v.id for v in self.metrics_run.values}
        for attribution in self.metrics_run.attributions:
            self.assertTrue(set(attribution.evidence["metrics"]) <= sources)
        self.assertTrue(all(h.status == "confirmed" for h in self.metrics_run.hypotheses))
        self.assertTrue(any(c.normalized_delta is not None for c in self.metrics_run.comparisons if c.metric == "audits_done"))

    def test_every_exported_demo_row_is_marked(self):
        lines = [line for line in render_markdown(self.metrics_run).splitlines() if line]
        self.assertTrue(all(line.startswith("ТЕСТОВЫЕ ДАННЫЕ · ") for line in lines))
        for field in ("values", "events", "comparisons", "attributions", "hypotheses", "knowledge", "recommendations"):
            self.assertTrue(all(item.is_synthetic for item in getattr(self.metrics_run, field)))
