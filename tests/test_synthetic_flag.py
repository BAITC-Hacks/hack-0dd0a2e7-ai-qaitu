from dataclasses import replace
import tempfile
import unittest

from qaitu.metrics_service import run_metrics
from tests.test_metrics_service import fixture


class SyntheticIsolationTests(unittest.TestCase):
    def test_mixed_values_fail_before_storage_or_model_call(self):
        from unittest.mock import patch
        run = fixture()
        mixed = run.values + [replace(run.values[0], id="real-value", is_synthetic=False)]
        with tempfile.TemporaryDirectory() as root, patch("qaitu.metrics_llm.enhance_attributions") as ai:
            with self.assertRaisesRegex(ValueError, "смешивать"):
                run_metrics(mixed, run.events, use_ai=True, storage_root=root)
            ai.assert_not_called()
            from pathlib import Path
            self.assertEqual(list(Path(root).iterdir()), [])
