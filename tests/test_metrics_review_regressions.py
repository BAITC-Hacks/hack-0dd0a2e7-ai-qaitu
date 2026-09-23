"""Regressions from the independent metrics review; synthetic data and mock API only."""
from dataclasses import asdict, replace
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from qaitu.metrics_llm import enhance_attributions, refine_recommendations
from qaitu.metrics_models import (Attribution, MetricSource, MetricValue, MetricsRun,
                                  PeriodComparison, Recommendation, StructuralEvent)
from qaitu.metrics_store import update_knowledge


def sample_run():
    values = [MetricValue(f"v{index}", "report_delay_days", number, "ДККМ", start, end, "month",
              MetricSource("SYNTH.xlsx", f"Задержка отчётов: {number} дня", "Свод", f"B{index + 1}"), True)
              for index, (number, start, end) in enumerate([
                  (3, "2022-01-01", "2022-01-31"), (3, "2022-02-01", "2022-02-28"),
                  (11, "2023-01-01", "2023-01-31"), (11, "2023-02-01", "2023-02-28")])]
    event = StructuralEvent("W1", "func_weakened", ["ДККМ"], "2022-12-23",
                            [{"file": "SYNTH.docx", "clause_id": "5.5.3", "quote": "Подготавливает квартальные отчёты."}],
                            metric_codes=["report_delay_days"], mechanism="Периодичность отчётности",
                            mechanism_confirmed=True, is_synthetic=True)
    comparison = PeriodComparison("C1", "report_delay_days", "ДККМ", "W1",
                                  {"value": 3, "n_points": 2}, {"value": 11, "n_points": 2}, 8, 800 / 3,
                                  method="simple", metric_ids=[v.id for v in values], is_synthetic=True)
    attribution = Attribution("A1", "W1", "C1", "После изменения задержка выросла: 3 → 11.", event.mechanism,
                              "medium", ["Механизм подтверждён человеком"], ["Возможное изменение формата отчётности"],
                              {"metrics": comparison.metric_ids}, opponent_outcome="upheld", is_synthetic=True)
    return MetricsRun("metrics_review_synth", True, values, [event], [comparison], [attribution])


def add_event(run, event_id, *, confidence="medium", same_mechanism=False):
    original = run.events[0]
    event = replace(original, id=event_id,
                    mechanism=original.mechanism if same_mechanism else f"Иная обязанность {event_id}")
    comparison = replace(run.comparisons[0], id=f"C_{event_id}", event_id=event_id)
    attribution = replace(run.attributions[0], id=f"A_{event_id}", event_id=event_id,
                          comparison_id=comparison.id, mechanism=event.mechanism, confidence=confidence)
    run.events.append(event)
    run.comparisons.append(comparison)
    run.attributions.append(attribution)


def mock_client(*responses):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=[
        SimpleNamespace(status="completed", output_text=json.dumps(response, ensure_ascii=False))
        for response in responses])))


def valid_draft():
    return {"attributions": [{"attribution_id": "A1", "claim": "После изменения задержка выросла: 3 → 11.",
              "mechanism": "Периодичность отчётности", "alternative_explanations": ["Возможное изменение формата отчётности"],
              "metric_ids": ["v0", "v1", "v2", "v3"], "clause_ids": ["5.5.3"]}]}


def valid_review():
    return {"reviews": [{"attribution_id": "A1", "outcome": "weakened",
                         "reasoning": "Наблюдаемое изменение требует проверки альтернатив.", "counter_evidence": []}]}


def sample_recommendation():
    return Recommendation("R1", "Проверить периодичность отчётности в ДККМ", "restore_function",
                          {"attributions": ["A1"]}, {"metric": "report_delay_days", "direction": "down"},
                          "medium", ["Возможная дополнительная нагрузка"], target_units=["ДККМ"], is_synthetic=True)


class KnowledgeIdentityRegressions(unittest.TestCase):
    def test_distinct_event_with_low_confidence_cannot_delete_another_case(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), TemporaryDirectory() as root:
                run = sample_run()
                add_event(run, "W2", confidence="low")
                if reverse:
                    run.attributions.reverse()
                knowledge = update_knowledge(run, root)
                self.assertEqual(sum(record.n_cases for record in knowledge), 1)
                self.assertEqual(knowledge[0].change_pattern, "func_weakened: периодичность отчётности")

    def test_distinct_supported_events_in_same_bucket_are_both_retained(self):
        with TemporaryDirectory() as root:
            run = sample_run()
            add_event(run, "W2")
            knowledge = update_knowledge(run, root)
            self.assertEqual(len(knowledge), 2)
            self.assertEqual(sum(record.n_cases for record in knowledge), 2)

    def test_reanalysis_without_comparisons_retracts_only_that_event(self):
        with TemporaryDirectory() as root:
            run = sample_run()
            add_event(run, "W2")
            update_knowledge(run, root)
            rerun = sample_run()
            rerun.events[0].mechanism_confirmed = False
            rerun.events[0].metric_codes = []
            rerun.comparisons = []
            rerun.attributions = []
            knowledge = update_knowledge(rerun, root)
            self.assertEqual(sum(record.n_cases for record in knowledge), 1)
            self.assertIn("w2", knowledge[0].change_pattern)

    def test_three_simultaneous_events_do_not_supply_three_diverse_cases(self):
        with TemporaryDirectory() as root:
            run = sample_run()
            add_event(run, "W2", same_mechanism=True)
            add_event(run, "W3", same_mechanism=True)
            record = update_knowledge(run, root)[0]
            self.assertEqual(record.n_cases, 1)
            self.assertEqual(record.diversity, 1)
            self.assertFalse(record.observed_effect["quantitative_allowed"])


class AIPublicationRegressions(unittest.TestCase):
    def assert_rules_retained(self, run, client):
        result, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions, client=client)
        self.assertEqual(result, run.attributions)
        self.assertTrue(warnings)
        self.assertNotIn("999999", " ".join(warnings))
        return result

    def test_unverified_target_department_is_not_published_as_ai_note(self):
        recommendation = sample_recommendation()
        data = {"recommendations": [{"recommendation_id": "R1",
                "action": "Передать всю отчетность Департаменту продаж",
                "risks": ["Возможная дополнительная нагрузка при согласовании"]}]}
        result, _ = refine_recommendations([recommendation], [], [], client=mock_client(data))
        self.assertEqual(result[0].action, recommendation.action)
        self.assertEqual(result[0].target_units, ["ДККМ"])
        self.assertNotIn("Департаменту продаж", json.dumps(asdict(result[0]), ensure_ascii=False))
        self.assertNotIn("ai_note", result[0].expected_effect)

    def test_invented_numbers_in_analyst_alternatives_restore_rules(self):
        run = sample_run()
        draft = valid_draft()
        draft["attributions"][0]["alternative_explanations"] = ["Возможно, штат вырос до 999999 сотрудников."]
        self.assert_rules_retained(run, mock_client(draft, valid_review()))

    def test_invented_numbers_in_opponent_fields_restore_rules(self):
        for field in ("reasoning", "counter_evidence"):
            with self.subTest(field=field):
                run = sample_run()
                review = valid_review()
                text = "Возможно, наняты 999999 сотрудников."
                review["reviews"][0][field] = [text] if field == "counter_evidence" else text
                self.assert_rules_retained(run, mock_client(valid_draft(), review))

    def test_invented_numbers_in_recommendation_risks_restore_rules(self):
        recommendation = sample_recommendation()
        data = {"recommendations": [{"recommendation_id": "R1", "action": recommendation.action,
                "risks": ["Возможно, потребуется нанять 999999 сотрудников."]}]}
        result, warnings = refine_recommendations([recommendation], [], [], client=mock_client(data))
        self.assertEqual(result, [recommendation])
        self.assertTrue(warnings)
        self.assertNotIn("999999", " ".join(warnings))


if __name__ == "__main__":
    unittest.main()
