import unittest

from qaitu.extractors import document_from_lines
from qaitu.linter import inspect_document_pack, lint_document, lint_documents


def document(lines, name="project.docx", period="after"):
    return document_from_lines(name, period, lines)


class DocumentLinterTests(unittest.TestCase):
    def test_empty_edition8_clause_with_exact_evidence(self):
        doc = document(["5.5.2. организует мониторинг качества внутреннего аудита;", "5.5.3. ;", "5.5.4. анализирует результаты непрерывного аудита."])
        findings = lint_document(doc)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "LNT-EMPTY")
        self.assertEqual(findings[0].sources[0].text, "5.5.3. ;")
        self.assertIn("5.5.3", findings[0].sources[0].locator)

    def test_correct_prohibition_reference_in_edition9_not_flagged(self):
        doc = document([
            "5.8. Главный аудитор и работники БВА не имеют права:",
            "5.8.1. выполнять функциональные обязанности, не связанные с деятельностью внутреннего аудита;",
            "5.8.2. руководить действиями работников других подразделений;",
            "5.10.2. доступ работников БВА ко всем документам, в части, не противоречащей п. 5.8.1 и 5.8.2;",
            "5.10.3. обеспечивает оборудованием;",
            "5.10.4. обеспечивает обучение;",
            "5.10.5. предоставление информации по запросу, в части, не противоречащей п. 5.8.1 и 5.8.2;",
        ])
        self.assertFalse(lint_document(doc))

    def test_semantic_candidate_has_target_and_heading_evidence(self):
        doc = document([
            "5.8. Работники БВА имеют право:",
            "5.8.1. запрашивать и получать доступ к документам;",
            "5.8.2. копировать документы;",
            "5.11.2. доступ к документам в части, не противоречащей п. 5.8.1 и 5.8.2;",
        ])
        findings = lint_document(doc)
        self.assertEqual([f.code for f in findings], ["LNT-REF-SEM"])
        self.assertEqual(len(findings[0].sources), 4)
        self.assertLess(findings[0].confidence, .7)
        self.assertIn("эвристический кандидат", findings[0].explanation)

    def test_reference_matching_is_exact_not_prefix(self):
        findings = lint_document(document(["5.1.1. выполняет проверку;", "6.1. согласно п. 5.1 настоящего Положения готовит отчёт."]))
        self.assertEqual([f.code for f in findings], ["LNT-REF"])
        self.assertIn("5.1", findings[0].title)

    def test_references_to_other_laws_not_internal_errors(self):
        doc = document(["1.1. В соответствии с п. 9.4 Трудового кодекса Российской Федерации.", "1.2. Согласно пункту 42.1 договора выполняется проверка."])
        self.assertFalse(lint_document(doc))

    def test_reference_ranges_check_interior_target(self):
        doc = document(["2.1. выполняет аудит;", "2.3. готовит отчёт;", "3.1. в соответствии с п. 2.1–2.3 настоящего Положения."])
        findings = lint_document(doc)
        self.assertEqual({f.code for f in findings}, {"LNT-NUM", "LNT-REF"})
        self.assertTrue(all("2.2" in f.title for f in findings))

    def test_duplicate_number_and_gap_have_both_sources(self):
        doc = document(["1.1. текст первого пункта;", "1.1. текст другого пункта;", "1.3. завершающий пункт."])
        findings = lint_document(doc)
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(f.code == "LNT-NUM" for f in findings))
        self.assertTrue(all(len(f.sources) >= 2 for f in findings))

    def test_partial_document_does_not_invent_edge_gaps(self):
        doc = document(["5.3.6. запрашивает информацию;", "5.3.7. организует контроль."])
        self.assertFalse(lint_document(doc))

    def test_toc_and_inherited_number_not_duplicates(self):
        doc = document(["1. Общие положения", "Текст вводного положения.", "1.1. Требования к работникам:", "а. трудовая дисциплина;", "б. сохранение тайны;", "Оглавление", "1. ОБЩИЕ ПОЛОЖЕНИЯ 1", "1.1. ТРЕБОВАНИЯ К РАБОТНИКАМ 3"])
        self.assertFalse(lint_document(doc))

    def test_two_documents_same_number_are_not_duplicates(self):
        before = document(["1.1. выполняет аудит."], "old.docx", "before")
        after = document(["1.1. проводит аудит."], "new.docx", "after")
        self.assertFalse(lint_documents([before, after]))

    def test_repeated_upload_dedupes_and_is_deterministic(self):
        doc = document(["1.1. ;"])
        self.assertEqual(lint_documents([doc]), lint_documents([doc, doc]))
        self.assertEqual(lint_document(doc), lint_document(doc))

    def test_single_document_never_invents_before_after_changes(self):
        doc = document(["1.1. ;"])
        result = inspect_document_pack([doc])
        self.assertFalse(result.findings)
        self.assertFalse(result.unit_changes)
        self.assertFalse(result.function_matches)
        self.assertFalse(result.matrix_rows)
        self.assertEqual(len(result.document_checks), 1)
        self.assertEqual(len(result.to_dict()["document_checks"]), 1)
        self.assertTrue(result.analysis_context["single_document_review"])
        self.assertEqual(result.coverage["documents_after"], 1)

    def test_single_document_preserves_extraction_warnings(self):
        doc = document(["1.1. ;"])
        doc.warnings = ["OCR не выполнен"]
        self.assertEqual(inspect_document_pack([doc]).warnings, doc.warnings)

    def test_single_document_requires_an_after_edition(self):
        for docs in ([], [document(["1.1. Проверка."], period="before")]):
            with self.assertRaises(ValueError):
                inspect_document_pack(docs)

    def test_unnumbered_document_reports_unavailable_checks(self):
        result = inspect_document_pack([document(["Права и обязанности определены отдельным документом."])])
        self.assertTrue(any("нет явно распознанной нумерации" in w for w in result.warnings))

    def test_every_finding_has_original_evidence(self):
        doc = document(["1.1. ;", "1.3. согласно п. 9.8 настоящего Положения выполняется аудит."])
        known = {s.id: s for s in doc.fragments}
        for finding in lint_document(doc):
            self.assertTrue(finding.sources)
            self.assertTrue(finding.assessment.evidence)
            for source in finding.sources:
                self.assertEqual(source, known[source.id])
                self.assertTrue(source.text.strip())
                self.assertIn("п.", source.locator)
