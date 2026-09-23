import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from qaitu import ai_agent
from qaitu.extractors import document_from_lines
from qaitu.models import AnalysisResult, Finding, Fragment


def response(payload=None, **kwargs):
    # Fixtures may include canonical ID lists to make test intent readable;
    # the actual model wire protocol only cites IDs inside evidence.
    wire = json.loads(json.dumps(payload or {"comparisons": [], "findings": []}))
    for collection in ("comparisons", "findings"):
        for item in wire.get(collection, []):
            if isinstance(item, dict):
                for field in ("before_source_ids", "after_source_ids", "source_ids"):
                    item.pop(field, None)
    values = dict(status="completed", output=[], output_text=json.dumps(wire, ensure_ascii=False))
    values.update(kwargs)
    return SimpleNamespace(**values)


class SemanticAgentTests(unittest.TestCase):
    def setUp(self):
        self.before = document_from_lines("old.docx", "before", ["Отдел финансов готовит отчет о расходах компании."])
        self.after = document_from_lines("new.docx", "after", ["Отдел финансов готовит отчет о расходах компании.", "Отдел аналитики готовит такой же отчет о расходах компании."])
        self.result = AnalysisResult([], [], [])
        self.all_sources = {source.id: source for source in self.before.fragments + self.after.fragments}
        def approve_all(**kwargs):
            return dict(status="completed", approved_comparisons=list(range(len(kwargs["comparisons"]))),
                        approved_findings=list(range(len(kwargs["findings"]))), reasons=[], rejected_count=0,
                        input_tokens=0, output_tokens=0, usage_available=True, requests_made=1)
        verifier_patch = patch.object(ai_agent, "verify_claims", side_effect=approve_all)
        self.verifier = verifier_patch.start()
        self.addCleanup(verifier_patch.stop)

    def comparison(self, kind="retained"):
        sources = self.before.fragments + self.after.fragments[:1]
        return dict(kind=kind, title="Сохранение подготовки отчетов", explanation="Подготовка отчета осталась у отдела финансов.",
                    before_source_ids=["B0001"], after_source_ids=["A0001"],
                    evidence=[dict(source_id=self.alias(source), quote=source.text) for source in sources],
                    recommendation="Уточнить периодичность отчета у владельца.")

    def risk(self):
        return dict(kind="duplicate", title="Возможный дубль отчета", explanation="Возможно совпадает предмет отчета двух отделов.", confidence=.7,
                    source_ids=[self.alias(source) for source in self.after.fragments],
                    evidence=[dict(source_id=self.alias(source), quote=source.text) for source in self.after.fragments],
                    recommendation="Проверить разделение областей ответственности.")

    def alias(self, source):
        if source.period == "before":
            return f"B{self.before.fragments.index(source) + 1:04d}"
        return f"A{self.after.fragments.index(source) + 1:04d}"

    def run_agent(self, api_response=None, **kwargs):
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = api_response or response()
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key", **kwargs)
        return client

    def test_full_catalog_single_request_and_strict_schema(self):
        client = self.run_agent(response({"comparisons": [self.comparison()], "findings": []}), model="gpt-4.1-mini")
        client.return_value.responses.create.assert_called_once()
        request = client.return_value.responses.create.call_args.kwargs
        payload = json.loads(request["input"][1]["content"])
        self.assertEqual(len(payload["sources"]), 3)
        for item in payload["sources"]:
            source = next(source for source in self.all_sources.values() if self.alias(source) == item["source_id"])
            self.assertEqual(item["text"], source.text)
            self.assertNotIn(source.id, request["input"][1]["content"])
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(request["model"], "gpt-4.1-mini")
        self.assertFalse(request["store"])
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        wire_properties = request["text"]["format"]["schema"]["properties"]
        self.assertNotIn("before_source_ids", wire_properties["comparisons"]["items"]["properties"])
        self.assertNotIn("after_source_ids", wire_properties["comparisons"]["items"]["properties"])
        self.assertNotIn("source_ids", wire_properties["findings"]["items"]["properties"])

        def check_schema(schema):
            if isinstance(schema, dict):
                if schema.get("type") == "object":
                    self.assertFalse(schema["additionalProperties"])
                    self.assertEqual(set(schema["required"]), set(schema["properties"]))
                for value in schema.values():
                    check_schema(value)
            elif isinstance(schema, list):
                for value in schema:
                    check_schema(value)
        check_schema(request["text"]["format"]["schema"])
        report = self.result.ai_review
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["covered_sources"], 3)
        self.assertEqual(report["sent_sources"], 3)
        self.assertEqual(report["completed_batches"], 1)
        self.assertIn("не исчерпывающий", report["summary"])
        self.assertEqual(len(report["comparisons"]), 1)
        self.assertEqual(report["comparisons"][0]["before_source_ids"], [self.before.fragments[0].id])
        self.assertEqual(report["comparisons"][0]["after_source_ids"], [self.after.fragments[0].id])
        self.assertIn("ai_review", self.result.to_dict())

    def test_invalid_literal_quote_and_wrong_period_are_rejected(self):
        bad_quote = self.comparison()
        bad_quote["evidence"][0]["quote"] = "Модель переписала исходную цитату"
        wrong_period = self.comparison()
        wrong_period["evidence"] = [dict(source_id="A0001", quote=self.after.fragments[0].text)]
        self.run_agent(response({"comparisons": [bad_quote, wrong_period], "findings": []}))
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.ai_review["rejected_items"], 2)
        self.assertEqual(self.result.ai_review["status"], "partial")
        self.assertEqual(self.result.ai_review["rejected_reasons"], {"quote_mismatch": 1, "missing_period": 1})

    def test_reasoning_effort_only_for_supported_mini_family(self):
        for model in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-mini-snapshot"):
            with self.subTest(model=model):
                client = self.run_agent(model=model)
                request = client.return_value.responses.create.call_args.kwargs
                if model.startswith("gpt-5.4-mini"):
                    self.assertEqual(request["reasoning"], {"effort": "low"})
                else:
                    self.assertNotIn("reasoning", request)

    def test_redundant_identifier_lists_are_outside_model_wire_schema(self):
        raw = json.dumps({"comparisons": [self.comparison()], "findings": []})
        self.run_agent(response(output_text=raw))
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.ai_review["rejected_reasons"], {"schema": 1})

    def test_unknown_extra_and_missing_evidence_are_rejected(self):
        unknown = self.comparison()
        unknown["evidence"][1]["source_id"] = "invented-id"
        missing = self.comparison()
        missing["evidence"] = []
        extra = self.comparison()
        extra["evidence"].append(dict(source_id="A0001", quote=self.after.fragments[0].text))
        self.run_agent(response({"comparisons": [unknown, missing, extra], "findings": []}))
        self.assertEqual(self.result.ai_review["rejected_items"], 3)
        self.assertEqual(self.result.ai_review["comparisons"], [])

    def test_joining_neighbouring_fragments_is_not_a_literal_quotation(self):
        joined = self.comparison()
        joined["evidence"][0]["quote"] = self.before.fragments[0].text + " ... " + self.after.fragments[1].text
        self.run_agent(response({"comparisons": [joined], "findings": []}))
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.ai_review["rejected_reasons"], {"quote_mismatch": 1})

    def test_canonical_ids_not_exposed_in_prompt_cannot_bypass_alias_validation(self):
        fabricated = self.comparison()
        fabricated["before_source_ids"] = [self.before.fragments[0].id]
        fabricated["evidence"][0]["source_id"] = self.before.fragments[0].id
        self.run_agent(response({"comparisons": [fabricated], "findings": []}))
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.ai_review["rejected_reasons"], {"unknown_source_id": 1})

    def test_loss_is_always_caveated_and_requires_both_periods(self):
        loss = self.comparison("possibly_lost")
        loss["title"] = "Закрепление отчетности"
        missing_after = self.comparison("possibly_lost")
        missing_after["after_source_ids"] = []
        missing_after["evidence"] = missing_after["evidence"][:1]
        self.run_agent(response({"comparisons": [loss, missing_after], "findings": []}))
        self.assertEqual(len(self.result.ai_review["comparisons"]), 1)
        accepted = self.result.ai_review["comparisons"][0]
        self.assertTrue(accepted["title"].startswith("Возможная потеря"))
        self.assertIn("не доказаны", accepted["explanation"])

    def test_role_heading_alone_cannot_establish_function_loss(self):
        self.before = document_from_lines("old.docx", "before", ["5.3. Директор направления внутреннего аудита:"])
        self.after = document_from_lines("new.docx", "after", ["5.3. Директоры департаментов и Директоры направлений ДИТААД и ДОА:"])
        self.all_sources = {source.id: source for source in self.before.fragments + self.after.fragments}
        comparison = self.comparison("possibly_lost")
        risk = dict(kind="loss", title="Возможная потеря функции директора", explanation="Прежняя должность не найдена.", confidence=.6,
                    evidence=comparison["evidence"], recommendation="Проверить старую обязанность и нового владельца.")
        self.run_agent(response({"comparisons": [comparison], "findings": [risk]}))
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.findings, [])
        self.assertEqual(self.result.ai_review["rejected_reasons"], {"structural_heading_not_function": 2})

    def test_added_requires_after_but_allows_no_before(self):
        addition = self.comparison("added")
        addition["before_source_ids"] = []
        addition["evidence"] = addition["evidence"][1:]
        self.run_agent(response({"comparisons": [addition], "findings": []}))
        self.assertEqual(len(self.result.ai_review["comparisons"]), 1)

    def test_risks_are_validated_and_deduplicated(self):
        risk = self.risk()
        invalid = self.risk()
        invalid["confidence"] = "0.7"
        self.run_agent(response({"comparisons": [self.comparison(), self.comparison()], "findings": [risk, risk, invalid]}))
        self.assertEqual(len(self.result.findings), 1)
        self.assertEqual(self.result.ai_review["accepted_findings"], 1)
        self.assertEqual(self.result.ai_review["rejected_items"], 1)
        self.assertEqual(len(self.result.ai_review["comparisons"]), 1)
        self.assertEqual(self.result.findings[0].sources, self.after.fragments)

    def test_existing_local_risk_is_not_duplicated(self):
        self.result.findings.append(Finding("duplicate", "Локальный риск", "Описание", .8, self.after.fragments))
        self.run_agent(response({"comparisons": [], "findings": [self.risk()]}))
        self.assertEqual(len(self.result.findings), 1)
        self.assertEqual(self.result.ai_review["accepted_findings"], 0)

    def test_bad_schema_refusal_and_incomplete_do_not_cover_sources(self):
        refusal = response(output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])])
        for api_response in [response(output_text="not json"), response({"comparisons": []}), response(status="incomplete"), refusal,
                             response(output_text='{"comparisons":[],"findings":[{"confidence":NaN}]}')]:
            with self.subTest(response=api_response):
                self.run_agent(api_response)
                self.assertEqual(self.result.ai_review["status"], "failed")
                self.assertEqual(self.result.ai_review["covered_sources"], 0)
                self.assertEqual(self.result.ai_review["completed_batches"], 0)
                self.assertEqual(self.result.ai_review["sent_sources"], 3)

    def test_api_error_is_generic_and_keeps_local_results(self):
        local = Finding("loss", "Локальный вывод", "Описание", .6, self.before.fragments)
        self.result.findings.append(local)
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.side_effect = RuntimeError("private-key private-document")
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
        self.assertEqual(self.result.ai_review["status"], "failed")
        self.assertEqual(self.result.findings, [local])
        self.assertNotIn("private", str(self.result.ai_review))
        self.assertNotIn("mock-key", str(self.result.ai_review))

    def test_api_http_errors_are_actionable_without_exception_content(self):
        for status, marker in [(401, "ключ"), (403, "разрешения"), (404, "Модель"), (429, "квоту"), (503, "недоступен")]:
            with self.subTest(status=status), patch("openai.OpenAI") as client:
                error = RuntimeError("secret-key request-text")
                error.status_code = status
                client.return_value.responses.create.side_effect = error
                ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
            self.assertIn(marker, self.result.ai_review["error"])
            self.assertNotIn("secret-key", str(self.result.ai_review))

    def test_partial_batches_count_only_successfully_completed_sources(self):
        def batch(ids):
            return {"sources": [dict(source_id=source_id, text=self.all_sources[source_id].text,
                                     period=self.all_sources[source_id].period, locator=self.all_sources[source_id].locator) for source_id in ids]}
        batches = [batch([self.before.fragments[0].id, self.after.fragments[0].id]), batch([self.after.fragments[1].id])]
        with patch.object(ai_agent, "_prepare_requests", return_value=(batches, self.all_sources, "retrieved_fragments")), patch("openai.OpenAI") as client:
            client.return_value.responses.create.side_effect = [response({"comparisons": [self.comparison()], "findings": []}), response(output_text="broken")]
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
        report = self.result.ai_review
        self.assertEqual(report["status"], "partial")
        self.assertEqual((report["completed_batches"], report["total_batches"]), (1, 2))
        self.assertEqual((report["covered_sources"], report["sent_sources"], report["total_sources"]), (2, 3, 3))
        self.assertEqual(len(report["comparisons"]), 1)

    def test_fallback_uses_bounded_existing_batches(self):
        with patch.object(ai_agent, "MAX_FULL_CATALOG_CHARS", 1):
            batches, sources, mode = ai_agent._prepare_requests([self.before, self.after], self.result)
        self.assertEqual(mode, "retrieved_fragments")
        self.assertLessEqual(len(batches), 8)
        self.assertEqual(len(sources), 3)
        for batch in batches:
            self.assertLessEqual(len(json.dumps(batch, ensure_ascii=False)), 60_000)
            transmitted, _ = ai_agent._transport_batch(batch, [self.before, self.after])
            self.assertLessEqual(len(json.dumps(transmitted, ensure_ascii=False)), 60_000)

    def test_oversized_warning_manifest_never_reaches_api(self):
        self.before.warnings.append("Страница без текста; " * 4000)
        with patch.object(ai_agent, "MAX_FULL_CATALOG_CHARS", 1), patch("openai.OpenAI") as client:
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
            client.return_value.responses.create.assert_not_called()
        self.assertEqual(self.result.ai_review["covered_sources"], 0)
        self.assertEqual(self.result.ai_review["sent_sources"], 0)
        self.assertNotEqual(self.result.ai_review["status"], "completed")
        self.assertIn("метаданные", self.result.ai_review["summary"])

    def test_no_key_and_exhausted_budget_make_no_requests(self):
        with patch("openai.OpenAI") as client:
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "")
            client.assert_not_called()
        self.assertEqual(self.result.ai_review["status"], "skipped")
        with patch("openai.OpenAI") as client, patch.object(ai_agent, "MAX_REVIEW_SECONDS", 0):
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
            client.return_value.responses.create.assert_not_called()
        self.assertEqual(self.result.ai_review["sent_sources"], 0)
        self.assertIn("временной бюджет", self.result.ai_review["error"])

    def test_usage_timeout_and_progress_are_reported(self):
        progress = Mock()
        with patch.object(ai_agent.time, "monotonic", side_effect=[100, 260]):
            client = self.run_agent(response(usage=SimpleNamespace(input_tokens=1234, output_tokens=123)), progress=progress)
        self.assertEqual(client.return_value.responses.create.call_args.kwargs["timeout"], 20)
        self.assertEqual(self.result.ai_review["input_tokens"], 1234)
        self.assertEqual(self.result.ai_review["output_tokens"], 123)
        self.assertTrue(self.result.ai_review["usage_available"])
        self.assertTrue(self.result.ai_review["usage_complete"])
        self.assertEqual(self.result.ai_review["usage_responses"], 1)
        self.assertTrue(progress.called)
        self.assertEqual(progress.call_args.args[:2], (1, 1))

    def test_token_usage_is_marked_incomplete_when_a_response_omits_usage(self):
        batch = {"sources": [dict(source_id=source.id, text=source.text, period=source.period, locator=source.locator) for source in self.all_sources.values()]}
        with patch.object(ai_agent, "_prepare_requests", return_value=([batch, batch], self.all_sources, "retrieved_fragments")), patch("openai.OpenAI") as client:
            client.return_value.responses.create.side_effect = [response(usage=SimpleNamespace(input_tokens=100, output_tokens=10)), response()]
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
        report = self.result.ai_review
        self.assertTrue(report["usage_available"])
        self.assertFalse(report["usage_complete"])
        self.assertEqual(report["usage_responses"], 1)
        self.assertEqual(report["requests_made"], 2)
        self.assertEqual(report["input_tokens"], 100)

    def test_ambiguous_ids_and_periods_fail_without_api_calls(self):
        bad = Fragment(self.before.fragments[0].id, "other", "Несовпадающий текст документа.", "п. 1", "before")
        self.before.fragments.append(bad)
        with patch("openai.OpenAI") as client:
            ai_agent.run_comparison_agent([self.before], [self.after], self.result, "mock-key")
            client.assert_not_called()
        self.assertEqual(self.result.ai_review["status"], "failed")

    def test_extraction_warning_prevents_complete_document_claim(self):
        self.before.warnings.append("OCR не выполнен")
        self.run_agent()
        self.assertEqual(self.result.ai_review["status"], "partial")
        self.assertIn("ограничения извлечения", self.result.ai_review["summary"])

    def test_verification_failure_excludes_ai_but_preserves_local_findings(self):
        local = Finding("loss", "Локальный вывод", "Локальная проверка", .7, self.before.fragments)
        self.result.findings.append(local)
        self.verifier.side_effect = ValueError("secret provider response")
        self.run_agent(response({"comparisons": [self.comparison()], "findings": [self.risk()]}))
        report = self.result.ai_review
        self.assertEqual(self.result.findings, [local])
        self.assertEqual(report["comparisons"], [])
        self.assertEqual(report["accepted_findings"], 0)
        self.assertEqual(report["verification_status"], "failed")
        self.assertEqual(report["verification_rejected"], 2)
        self.assertEqual(report["status"], "partial")
        self.assertNotIn("secret", str(report))

    def test_verification_filters_claims_and_includes_usage(self):
        self.verifier.side_effect = None
        self.verifier.return_value = dict(status="completed", approved_comparisons=[1], approved_findings=[],
                                          reasons=["C1: вывод не следует из источников", "R1: разные области ответственности"],
                                          rejected_count=2, input_tokens=50, output_tokens=5, usage_available=True, requests_made=1)
        self.run_agent(response({"comparisons": [self.comparison(), self.comparison("changed")], "findings": [self.risk()]},
                                usage=SimpleNamespace(input_tokens=100, output_tokens=10)))
        report = self.result.ai_review
        self.assertEqual([item["kind"] for item in report["comparisons"]], ["changed"])
        self.assertEqual(self.result.findings, [])
        self.assertEqual(report["verification_rejected"], 2)
        self.assertEqual(report["verification_status"], "completed")
        self.assertEqual(report["status"], "partial")
        self.assertEqual((report["input_tokens"], report["output_tokens"]), (150, 15))
        self.assertEqual(report["requests_made"], 2)
        self.assertEqual(report["completed_batches"], 1)
        self.assertTrue(report["usage_complete"])
        self.assertLessEqual(self.verifier.call_args.kwargs["timeout"], 90)

    def test_verification_budget_exhaustion_drops_unverified_candidates(self):
        with patch.object(ai_agent.time, "monotonic", side_effect=[100, 100, 281]):
            self.run_agent(response({"comparisons": [self.comparison()], "findings": []}))
        self.verifier.assert_not_called()
        self.assertEqual(self.result.ai_review["verification_status"], "budget_exhausted")
        self.assertEqual(self.result.ai_review["comparisons"], [])
        self.assertEqual(self.result.ai_review["requests_made"], 1)

    def test_no_candidates_skip_semantic_verification(self):
        self.run_agent(response())
        self.verifier.assert_not_called()
        self.assertEqual(self.result.ai_review["verification_status"], "not_needed")


if __name__ == "__main__":
    unittest.main()
