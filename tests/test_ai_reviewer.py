import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qaitu import ai_reviewer
from qaitu.extractors import document_from_lines
from qaitu.models import AnalysisResult, Finding


def response(findings=None, raw=None, **kwargs):
    return SimpleNamespace(status="completed", output=[], output_text=raw if raw is not None else json.dumps({"findings": findings or []}), **kwargs)


class AiReviewerTests(unittest.TestCase):
    def setUp(self):
        self.before = document_from_lines("old.docx", "before", ["Отдел финансов готовит отчет об операционных расходах."])
        self.after = document_from_lines("new.docx", "after", ["Отдел финансов формирует отчет об операционных расходах.", "Отдел аналитики составляет тот же отчет об операционных расходах."])
        self.result = AnalysisResult([], [], [])

    def finding(self, kind="duplicate", sources=None, **kwargs):
        sources = sources if sources is not None else self.after.fragments
        return dict(kind=kind, title="Возможный дубль отчета", explanation="Два отдела формируют одинаковый отчет.", confidence=0.7,
                    source_ids=[source.id for source in sources], evidence={source.id: source.text for source in sources},
                    recommendation="Уточнить границы ответственности отделов.", **kwargs)

    def run_review(self, value):
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = value
            findings = ai_reviewer.review_with_llm([self.before], [self.after], self.result, "test-key")
            return findings, client

    def test_valid_finding_is_grounded_and_api_storage_disabled(self):
        findings, client = self.run_review(response([self.finding()]))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].sources, self.after.fragments)
        self.assertFalse(client.return_value.responses.create.call_args.kwargs["store"])

    def test_llm_self_report_is_labelled_capped_and_never_confirmed(self):
        value = self.finding()
        value["confidence"] = 1.0
        findings, _ = self.run_review(response([value]))
        self.assertEqual(findings[0].confidence, .89)
        self.assertEqual(findings[0].assessment.method, "llm-self-report")
        self.assertEqual(findings[0].assessment.metrics["reported_score"], 1.0)
        self.assertTrue(findings[0].assessment.limitations)

    def test_exhausted_time_budget_makes_no_api_request(self):
        with patch("openai.OpenAI") as client, patch.object(ai_reviewer, "MAX_REVIEW_SECONDS", 0):
            findings = ai_reviewer.review_with_llm([self.before], [self.after], self.result, "test-key")
        self.assertEqual(findings, [])
        client.return_value.responses.create.assert_not_called()
        self.assertTrue(any("исчерпан бюджет" in warning for warning in self.result.warnings))

    def test_request_timeout_is_capped_by_remaining_time_and_retries_disabled(self):
        for elapsed, expected in [(0, 45), (160, 20)]:
            with self.subTest(elapsed=elapsed), patch("openai.OpenAI") as client, patch.object(ai_reviewer.time, "monotonic", side_effect=[100, 100 + elapsed]):
                client.return_value.responses.create.return_value = response()
                ai_reviewer.review_with_llm([self.before], [self.after], self.result, "test-key")
            self.assertEqual(client.return_value.responses.create.call_args.kwargs["timeout"], expected)
            self.assertEqual(client.call_args.kwargs["max_retries"], 0)

    def test_reported_api_token_usage_is_visible(self):
        api_response = response(usage=SimpleNamespace(input_tokens=100, output_tokens=12))
        self.run_review(api_response)
        self.assertTrue(any("100 входных и 12 выходных токенов" in warning for warning in self.result.warnings))
        self.assertTrue(any("отправлено запросов 1" in warning for warning in self.result.warnings))

    def test_missing_api_usage_is_explicitly_unavailable(self):
        self.run_review(response())
        self.assertTrue(any("не вернул данные о расходе токенов" in warning for warning in self.result.warnings))

    def test_unknown_and_unquoted_sources_are_discarded(self):
        unknown = self.finding()
        unknown["source_ids"].append("invented")
        bad_quote = self.finding()
        bad_quote["evidence"][self.after.fragments[0].id] = "Несуществующая цитата"
        self.assertEqual(self.run_review(response([unknown, bad_quote]))[0], [])

    def test_invalid_types_and_nonfinite_confidence_fail_closed(self):
        invalid = []
        for confidence in [None, "0.5", True, -1, 3, 10 ** 1000]:
            value = self.finding()
            value["confidence"] = confidence
            invalid.append(value)
        invalid.extend([None, [], {"kind": "other"}, {"kind": []}])
        self.assertEqual(self.run_review(response(invalid))[0], [])
        self.assertEqual(self.run_review(response(raw='{"findings":[{"confidence":NaN}]}'))[0], [])

    def test_loss_requires_both_periods_and_duplicates_require_two_after_sources(self):
        invalid = [self.finding("loss", self.before.fragments), self.finding("duplicate", self.before.fragments + self.after.fragments[:1])]
        self.assertEqual(self.run_review(response(invalid))[0], [])
        valid = self.finding("loss", self.before.fragments + self.after.fragments[:1])
        findings = self.run_review(response([valid]))[0]
        self.assertEqual(len(findings), 1)
        self.assertIn("не доказано", findings[0].explanation)

    def test_base_and_batch_duplicates_are_not_added_again(self):
        self.result.findings = [Finding("duplicate", "Существующий вывод", "Уже обнаружено", .6, self.after.fragments)]
        self.assertEqual(self.run_review(response([self.finding(), self.finding()]))[0], [])

    def test_malformed_refusal_and_incomplete_responses_preserve_base(self):
        refusal = response()
        refusal.output = [SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="Нет")])]
        incomplete = response()
        incomplete.status = "incomplete"
        for value in [response(raw="not json"), response(raw="[]"), refusal, incomplete]:
            with self.subTest(value=value):
                self.assertEqual(self.run_review(value)[0], [])
        self.assertTrue(any("отклонён" in warning for warning in self.result.warnings))

    def test_api_error_is_safe_and_does_not_escape(self):
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.side_effect = RuntimeError("secret-key request body")
            findings = ai_reviewer.review_with_llm([self.before], [self.after], self.result, "test-key")
        self.assertEqual(findings, [])
        self.assertNotIn("secret-key", " ".join(self.result.warnings))
        self.assertTrue(any("ошибка API" in warning for warning in self.result.warnings))

    def test_large_input_is_batched_without_truncating_quotes(self):
        before = document_from_lines("big-old.docx", "before", [f"{i}. Отдел готовит отчет по объекту {i}: " + "контроль операции " * 170 for i in range(70)])
        after = document_from_lines("big-new.docx", "after", [f"{i}. Отдел готовит отчет по объекту {i}: " + "контроль операции " * 170 for i in range(70)])
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = response()
            findings = ai_reviewer.review_with_llm([before], [after], self.result, "test-key")
        calls = client.return_value.responses.create.call_args_list
        self.assertGreater(len(calls), 1)
        self.assertLessEqual(len(calls), ai_reviewer.MAX_BATCHES)
        self.assertEqual(findings, [])
        source_map = {fragment.id: fragment for fragment in before.fragments + after.fragments}
        for call in calls:
            encoded = call.kwargs["input"][1]["content"]
            self.assertLessEqual(len(encoded), ai_reviewer.MAX_BATCH_CHARS)
            payload = json.loads(encoded)
            self.assertEqual({item["period"] for item in payload["sources"]}, {"before", "after"})
            for source in payload["sources"]:
                self.assertEqual(source["text"], source_map[source["source_id"]].text)
        self.assertTrue(any("неполное" in warning for warning in self.result.warnings))

    def test_source_from_another_batch_cannot_be_cited(self):
        unprovided = document_from_lines("elsewhere.docx", "after", ["Другой отдел составляет независимый отчет по расходам."]).fragments[0]
        finding = self.finding(sources=[self.after.fragments[0], unprovided])
        self.assertEqual(self.run_review(response([finding]))[0], [])


if __name__ == "__main__":
    unittest.main()
