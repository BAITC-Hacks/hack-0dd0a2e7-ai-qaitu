import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qaitu.ai_verifier import ERROR_MESSAGE, verify_claims
from qaitu import ai_verifier
from qaitu.models import Finding, Fragment


def response(reviews, **kwargs):
    values = dict(status="completed", output=[], output_text=json.dumps({"reviews": reviews}, ensure_ascii=False),
                  usage=SimpleNamespace(input_tokens=100, output_tokens=30))
    values.update(kwargs)
    return SimpleNamespace(**values)


def verdict(claim_id, approved=True):
    return {"claim_id": claim_id, "verdict": "approve" if approved else "reject",
            "reason": "Сохранение обязанности подтверждено обеими цитатами." if approved else "Заголовки ролей не доказывают потери конкретной обязанности."}


class SemanticVerifierTests(unittest.TestCase):
    def setUp(self):
        self.old = Fragment("before:canonical:1", "old.docx", "Отдел качества готовит отчёт о проверках компании.", "п. 1", "before")
        self.new = Fragment("after:canonical:1", "new.docx", "Отдел качества готовит отчёт о проверках компании.", "п. 2", "after")
        self.context = Fragment("after:canonical:2", "new.docx", "Директор отдела качества отвечает за контроль отчётности.", "п. 3", "after")
        self.sources = {source.id: source for source in (self.old, self.new, self.context)}
        self.comparisons = [{"kind": "retained", "title": "Обязанность подготовки отчета сохранена",
            "explanation": "Отдел качества готовит тот же отчет в обеих редакциях.",
            "before_source_ids": [self.old.id], "after_source_ids": [self.new.id],
            "evidence": [{"source_id": self.old.id, "quote": self.old.text}, {"source_id": self.new.id, "quote": self.new.text}],
            "recommendation": "Сверить периодичность."}]
        self.finding = Finding("loss", "Возможная потеря прежней роли", "Старая должность отсутствует в новом заголовке.", .5,
                               [self.old, self.new], "Проверить штатное расписание.")
        self.client = Mock()

    def run_verifier(self, reviews, findings=None, contexts=None, model="gpt-5.4-mini", **kwargs):
        self.client.responses.create.return_value = response(reviews, **kwargs)
        return verify_claims(self.client, model, self.comparisons, findings or [], self.sources,
                             contexts or [], timeout=30)

    def test_selective_approval_does_not_rewrite_claims_and_tracks_usage(self):
        original = json.dumps(self.comparisons, ensure_ascii=False)
        result = self.run_verifier([verdict("C0"), verdict("F0", False)], [self.finding], [self.context])
        self.assertEqual(result["approved_comparisons"], [0])
        self.assertEqual(result["approved_findings"], [])
        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["input_tokens"], 100)
        self.assertTrue(result["usage"]["usage_available"])
        self.assertEqual(result["requests_made"], 1)
        self.assertEqual(json.dumps(self.comparisons, ensure_ascii=False), original)
        self.assertNotIn("comparisons", result)
        request = self.client.responses.create.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["timeout"], 30)
        self.assertTrue(request["text"]["format"]["strict"])
        payload = json.loads(request["input"][1]["content"])
        self.assertEqual(len(payload["sources"]), 3)
        self.assertEqual({source["text"] for source in payload["sources"]}, {self.old.text, self.new.text, self.context.text})
        self.assertEqual(len(payload["documents"]), 2)
        self.assertNotIn(self.old.id, request["input"][1]["content"])

    def test_every_claim_must_have_exactly_one_known_verdict(self):
        invalid = [[], [verdict("unknown")], [verdict("C0"), verdict("C0")]]
        for reviews in invalid:
            with self.subTest(reviews=reviews), self.assertRaisesRegex(ValueError, ERROR_MESSAGE):
                self.run_verifier(reviews)
        with self.assertRaises(ValueError):
            self.run_verifier([verdict("C0"), verdict("C0")], [self.finding])

    def test_missing_findings_verdict_fails_whole_response(self):
        with self.assertRaises(ValueError):
            self.run_verifier([verdict("C0")], [self.finding])

    def test_empty_claims_make_no_request(self):
        result = verify_claims(self.client, "gpt-4.1-mini", [], [], self.sources, [], 30)
        self.assertEqual(result["approved_comparisons"], [])
        self.assertEqual(result["requests_made"], 0)
        self.client.responses.create.assert_not_called()

    def test_invalid_or_unknown_cited_sources_fail_before_api(self):
        self.comparisons[0]["evidence"][0]["source_id"] = "invented"
        with self.assertRaises(ValueError):
            self.run_verifier([verdict("C0")])
        self.client.responses.create.assert_not_called()

    def test_required_source_text_is_never_truncated_to_fit(self):
        with patch.object(ai_verifier, "MAX_PAYLOAD_CHARS", 200), self.assertRaises(ValueError) as error:
            self.run_verifier([verdict("C0")])
        self.assertEqual(error.exception.requests_made, 0)
        self.client.responses.create.assert_not_called()

    def test_extra_context_can_be_omitted_without_cutting_cited_text(self):
        huge = Fragment("after:huge", "context.docx", "длинный контекст " * 10_000, "п. 4", "after")
        self.sources[huge.id] = huge
        with patch.object(ai_verifier, "MAX_PAYLOAD_CHARS", 2000):
            result = self.run_verifier([verdict("C0")], contexts=[huge])
        payload = json.loads(self.client.responses.create.call_args.kwargs["input"][1]["content"])
        self.assertEqual(result["context_omitted"], 1)
        self.assertEqual(payload["additional_context_omitted"], 1)
        self.assertEqual({source["text"] for source in payload["sources"]}, {self.old.text, self.new.text})
        self.assertLessEqual(len(self.client.responses.create.call_args.kwargs["input"][1]["content"]), 2000)

    def test_provider_failure_exposes_no_credentials_or_provider_text(self):
        self.client.responses.create.side_effect = RuntimeError("private-api-key and confidential request body")
        with self.assertRaises(ValueError) as error:
            verify_claims(self.client, "gpt-5.4-mini", self.comparisons, [], self.sources, [], 30)
        self.assertEqual(str(error.exception), ERROR_MESSAGE)
        self.assertTrue(error.exception.__suppress_context__)
        self.assertEqual(error.exception.requests_made, 1)
        self.assertFalse(error.exception.usage["usage_available"])
        self.client.responses.create.assert_called_once()

    def test_malformed_incomplete_and_refusal_responses_fail_closed(self):
        cases = [response([], output_text="not json"), response([verdict("C0")], status="incomplete"),
                 response([verdict("C0")], output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])]),
                 response([], output_text='{"reviews":[],"rewritten_claim":"trust me"}')]
        for value in cases:
            self.client.responses.create.return_value = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_claims(self.client, "gpt-5.4-mini", self.comparisons, [], self.sources, [], 30)

    def test_invalid_verdict_fields_fail_closed(self):
        cases = [{"claim_id": "C0", "verdict": "approve", "reason": ""},
                 {"claim_id": "C0", "verdict": "maybe", "reason": "Не уверен"},
                 {"claim_id": "C0", "verdict": [], "reason": "Нет"},
                 {"claim_id": "C0", "verdict": "approve", "reason": "Да", "new_quote": "нет"}]
        for review in cases:
            with self.subTest(review=review), self.assertRaises(ValueError):
                self.run_verifier([review])

    def test_unknown_usage_is_not_reported_as_known_zero_and_old_model_has_no_reasoning(self):
        result = self.run_verifier([verdict("C0")], model="gpt-4.1-mini", usage=None)
        self.assertFalse(result["usage_available"])
        self.assertNotIn("reasoning", self.client.responses.create.call_args.kwargs)

    def test_exhausted_timeout_makes_no_request(self):
        for timeout in (0, -1, float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                verify_claims(self.client, "gpt-5.4-mini", self.comparisons, [], self.sources, [], timeout)
        self.client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
