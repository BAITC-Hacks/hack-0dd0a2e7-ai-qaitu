import json
import unittest
from dataclasses import replace

from qaitu.analyzer import _action_stems, _similarity, _tokens, analyze_documents
from qaitu.confidence import assess_result, assessment, confidence_counts, confidence_level
from qaitu.demo import demo_documents
from qaitu.models import AnalysisResult, Document, Finding, Fragment, Function, MatrixRow
from qaitu.reporting import collect_sources, markdown_report


def function(key, text, *, unit="ДККМ", period="after", **kwargs):
    source = Fragment(key, period + ".docx", text, "п. " + key, period)
    return Function(key, unit, text, source, **kwargs)


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.old = function("5.6.2", "формировать группы контроля качества", period="before", norm_type="right")
        self.new = function("5.5.3", "готовит отчеты о доходах", unit="ДНМ")

    def loss(self, after=None, *, complete=False, warnings=None, old=None):
        old = old or self.old
        after = [self.new] if after is None else after
        row = MatrixRow("lost", old.text, [old], [], "lost", old.norm_type)
        finding = Finding("loss", "Прямое право не найдено", old.text, .5, [old.source], matrix_row_id=row.id)
        result = AnalysisResult([], [], [finding], matrix_rows=[row] + [
            MatrixRow(f.id, f.text, [], [f], "new", f.norm_type) for f in after])
        self.score(result, [old], after, complete=complete, warnings=warnings)
        return result, finding.assessment

    def score(self, result, before, after, *, complete=False, warnings=None):
        assess_result(result, [Document("before.docx", "before", [f.source for f in before])],
                      [Document("after.docx", "after", [f.source for f in after], warnings=warnings or [])],
                      after_complete=complete, similarity=_similarity, tokens=_tokens, actions=_action_stems)

    def test_bands_and_rounding_are_consistent_and_never_100(self):
        for score, expected in [(0, "weak"), (.39, "weak"), (.40, "review"), (.69, "review"),
                                (.70, "high"), (.89, "high"), (.90, "very_high"), (1, "very_high")]:
            with self.subTest(score=score):
                self.assertEqual(confidence_level(score), expected)
                self.assertLessEqual(assessment(score).score, .99)
        self.assertEqual(confidence_level(.896), "very_high")

    def test_full_package_is_not_inferred_from_clean_extraction(self):
        result, value = self.loss()
        self.assertLessEqual(value.score, .89)
        self.assertEqual(value.metrics["package_completeness"], "Не подтверждена")
        self.assertFalse(result.analysis_context["after_complete_user_declared"])
        self.assertEqual(value.metrics["owners_checked"], 1)
        self.assertEqual(value.metrics["assignments_checked"], 1)
        self.assertEqual(value.evidence, [self.new.source])
        self.assertEqual(value.metrics["nearest_text_similarity"], round(_similarity(self.old.text, self.new.text), 4))

    def test_declared_complete_clean_known_package_can_be_very_high_not_certain(self):
        _, value = self.loss(complete=True)
        self.assertGreaterEqual(value.score, .90)
        self.assertLess(value.score, 1)
        self.assertEqual(value.metrics["package_completeness"], "Заявлена пользователем")

    def test_unknown_owner_text_is_searched_and_caps_loss(self):
        unknown = replace(self.new, owner_known=False, unit="Владелец не установлен")
        _, value = self.loss([unknown], complete=True)
        self.assertLessEqual(value.score, .89)
        self.assertEqual(value.metrics["owners_checked"], 0)
        self.assertEqual(value.metrics["assignments_checked"], 1)
        self.assertEqual(value.metrics["unknown_owners_after"], 1)
        self.assertEqual(value.evidence, [unknown.source])

    def test_missing_owner_old_empty_after_and_ocr_warning_reduce_scores(self):
        _, unknown = self.loss(old=replace(self.old, owner_known=False), complete=True)
        _, empty = self.loss([], complete=True)
        _, ocr = self.loss(complete=True, warnings=["Страница без текста; требуется OCR"])
        self.assertLessEqual(unknown.score, .39)
        self.assertLessEqual(empty.score, .39)
        self.assertLessEqual(ocr.score, .69)

    def test_plausible_transfer_reduces_confidence_and_preserves_sources(self):
        _, base = self.loss(complete=True)
        transferred = replace(self.old, id="transfer", source=replace(self.old.source, id="transfer", period="after"), unit="ДНМ")
        _, value = self.loss([self.new, transferred], complete=True)
        self.assertTrue(value.metrics["possible_transfer"])
        self.assertLess(value.score, base.score)
        self.assertLessEqual(value.score, .69)
        self.assertIn(transferred.source, value.evidence)

    def test_possible_split_is_explained_and_caps_confidence(self):
        old = function("old", "формирует реестр договоров и согласовывает реестр договоров", period="before")
        parts = [function("part1", "формирует реестр договоров", unit="ДНМ"),
                 function("part2", "согласовывает реестр договоров", unit="ДОА")]
        _, value = self.loss(parts, complete=True, old=old)
        self.assertTrue(value.metrics["possible_split"])
        self.assertLessEqual(value.score, .69)

    def test_right_and_prohibition_are_not_equivalent_support(self):
        same = replace(self.old, id="after", source=replace(self.old.source, period="after"))
        _, matching = self.loss([same])
        _, other_norm = self.loss([replace(same, norm_type="prohibition")])
        self.assertLess(other_norm.metrics["strongest_weighted_match"], matching.metrics["strongest_weighted_match"])
        self.assertEqual(other_norm.metrics["nearest_norm_type"], "prohibition")

    def test_matched_rows_explain_preservation_transfer_and_split(self):
        old = replace(self.new, id="old", source=replace(self.new.source, period="before"))
        for status in ("preserved", "moved", "changed"):
            after = replace(self.new, unit="ДОА" if status == "moved" else old.unit)
            row = MatrixRow("row", old.text, [old], [after], status)
            result = AnalysisResult([], [], [], matrix_rows=[row])
            self.score(result, [old], [after], complete=True)
            self.assertEqual(row.assessment.score, .99)
            self.assertEqual(row.assessment.metrics["match_coverage"], 1)
            row.notes = ["Составная прежняя функция разделена"]
            self.score(result, [old], [after], complete=True)
            self.assertLessEqual(row.assessment.score, .89)

    def test_duplicate_different_scope_is_lower_than_same_scope(self):
        scores = []
        for scope in ("единый процесс", "другая область", ""):
            a = replace(self.new, scope="единый процесс")
            b = replace(self.new, id="second", unit="ДОА", source=replace(self.new.source, id="second"), scope=scope)
            row = MatrixRow("dup", a.text, [], [a, b], "new", candidate_overlap=True)
            finding = Finding("duplicate", "Дубль", "", .65, [a.source, b.source], matrix_row_id=row.id)
            result = AnalysisResult([], [], [finding], matrix_rows=[row])
            self.score(result, [self.old], [a, b], complete=True)
            scores.append(finding.confidence)
            self.assertEqual(finding.assessment.metrics["text_similarity"], 1)
        self.assertGreater(scores[0], scores[1])
        self.assertLessEqual(scores[1], .69)
        self.assertLessEqual(scores[2], .89)

    def test_end_to_end_demo_conflict_and_export_keep_explanations(self):
        before, after = demo_documents()
        result = analyze_documents(before, after)
        self.assertEqual({f.kind for f in result.findings}, {"loss", "duplicate", "conflict"})
        for finding in result.findings:
            self.assertEqual(finding.confidence, finding.assessment.score)
            self.assertTrue(finding.assessment.reasons)
            self.assertLess(finding.confidence, 1)
        conflict = next(f for f in result.findings if f.kind == "conflict")
        self.assertIn("same_owner", conflict.assessment.metrics)
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
        report = markdown_report(result)
        self.assertIn("Уверенность алгоритма", report)
        self.assertIn("не математическая вероятность", report)
        self.assertEqual(sum(sum(levels.values()) for levels in confidence_counts(result.findings).values()), len(result.findings))
        source_ids = {s.id for s in collect_sources(result)}
        self.assertTrue(all(s.id in source_ids for f in result.findings for s in f.assessment.evidence))

    def test_shared_clause_is_not_independent_duplicate_evidence(self):
        a = replace(self.new, scope="единый процесс")
        b = replace(a, id="second-owner", unit="ДОА")
        row = MatrixRow("common", a.text, [], [a, b], "new", candidate_overlap=True)
        finding = Finding("duplicate", "Общий пункт", "", .65, [a.source], matrix_row_id=row.id)
        result = AnalysisResult([], [], [finding], matrix_rows=[row])
        self.score(result, [self.old], [a, b], complete=True)
        self.assertLessEqual(finding.confidence, .69)
        self.assertFalse(finding.assessment.metrics["distinct_sources"])
        self.assertEqual(finding.assessment.evidence, [a.source])

    def test_assessment_only_source_is_included_in_export_catalog(self):
        result, value = self.loss()
        result.matrix_rows = []
        self.assertIn(self.new.source, collect_sources(result))
        self.assertIn(self.new.source.text, markdown_report(result))


if __name__ == "__main__":
    unittest.main()
