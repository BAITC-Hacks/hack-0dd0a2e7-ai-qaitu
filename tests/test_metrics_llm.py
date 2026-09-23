import unittest
import json
from types import SimpleNamespace
from unittest.mock import Mock

from qaitu.metrics_llm import extract_metrics, enhance_attributions, refine_recommendations
from tests.test_metrics_service import fixture


def client_with(text, status="completed"):
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(status=status, output_text=text))))
    return client


class LLMTests(unittest.TestCase):
    def test_extraction_is_explicit_structured_and_not_stored(self):
        client = client_with('{"values":[],"ambiguous":[],"unknown_metrics":[]}')
        self.assertEqual(extract_metrics([], [], client=client)["values"], [])
        kwargs = client.responses.create.call_args.kwargs
        self.assertFalse(kwargs["store"])
        self.assertTrue(kwargs["text"]["format"]["strict"])
        self.assertEqual(kwargs["timeout"], 45)

    def test_provider_failure_does_not_leak_secret_or_drop_rules(self):
        client = client_with("bad")
        client.responses.create.side_effect = RuntimeError("sk-sensitive-secret report-fragment")
        run = fixture()
        result, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions, client=client)
        self.assertEqual(result, run.attributions)
        self.assertNotIn("sk-sensitive", str(warnings))
        self.assertTrue(warnings)

    def test_incomplete_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "ИИ-этап"):
            extract_metrics([], [], client=client_with("{}", "incomplete"))

    def test_unknown_claim_or_causal_language_not_published(self):
        run = fixture()
        for text in ('{"attributions":[{"attribution_id":"invented"}]}',
                     '{"attributions":[{"attribution_id":"a1","claim":"Изменение привело к росту","mechanism":"reporting","alternative_explanations":["найм"],"metric_ids":["mv1"],"clause_ids":["5.5.3"]}]}'):
            result, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions, client=client_with(text))
            self.assertEqual(result, run.attributions)
            self.assertTrue(warnings)


class ReviewedAttributionTests(unittest.TestCase):
    def test_independent_opponent_can_only_lower_rule_confidence(self):
        run = fixture()
        run.events[0].metric_codes = ["report_delay_days"]
        draft = {"attributions": [{"attribution_id": "a1", "claim": "После изменения задержка выросла: 3 → 11.",
                 "mechanism": "Периодичность отчётности", "alternative_explanations": ["Возможное изменение формата отчётности"],
                 "metric_ids": ["mv1"], "clause_ids": ["5.5.3"]}]}
        review = {"reviews": [{"attribution_id": "a1", "outcome": "upheld", "reasoning": "Проверены источники, альтернативы сохраняются.", "counter_evidence": []}]}
        client = client_with("")
        client.responses.create.side_effect = [SimpleNamespace(status="completed", output_text=json.dumps(item)) for item in (draft, review)]
        result, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions, client=client)
        self.assertEqual(warnings, [])
        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(result[0].method, "llm+rules")
        self.assertEqual(result[0].confidence, "medium")
        self.assertIn("Найм", result[0].alternative_explanations)

    def test_invented_numeric_effect_does_not_reach_opponent(self):
        run = fixture()
        draft = {"attributions": [{"attribution_id": "a1", "claim": "После изменения показатель вырос на 999999%.",
                 "mechanism": "Периодичность отчётности", "alternative_explanations": ["Найм"],
                 "metric_ids": ["mv1"], "clause_ids": ["5.5.3"]}]}
        client = client_with(json.dumps(draft))
        result, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions, client=client)
        self.assertEqual(client.responses.create.call_count, 1)
        self.assertEqual(result, run.attributions)
        self.assertTrue(warnings)
