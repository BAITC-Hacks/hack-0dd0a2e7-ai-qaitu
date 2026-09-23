from contextlib import closing
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest

from docx import Document as WordDocument
from pypdf import PdfReader

from qaitu.analyzer import analyze_documents
from qaitu.approval_export import approval_docx, approval_pdf
from qaitu.demo import demo_documents
from qaitu.extractors import document_from_lines
from qaitu.linter import inspect_document_pack, lint_documents
from qaitu.models import Finding
from qaitu.review import ReviewStore, current_decisions, finding_id, gate_status, package_id, review_items


class ReviewChecklistTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "local" / "review.sqlite3"
        self.store = ReviewStore(self.path)
        before, after = demo_documents()
        self.result = analyze_documents(before, after, after_complete=True)
        self.package = package_id(self.result)
        self.finding = self.result.findings[0]

    def save(self, **kwargs):
        fields = {"status": "intentional", "actor": "Эксперт", "comment": "Рассмотрено по исходному пункту."}
        fields.update(kwargs)
        return self.store.save(self.package, self.finding, **fields)

    def test_untouched_has_red_gate(self):
        gate = gate_status(self.result, [])
        self.assertEqual(gate["color"], "red")
        self.assertTrue(gate["blocker_ids"])

    def test_reading_new_pack_does_not_create_database(self):
        self.assertEqual(self.store.history(self.package), [])
        self.assertFalse(self.path.exists())

    def test_blank_actor_or_comment_cannot_close(self):
        for fields in ({"comment": " "}, {"actor": " "}, {"status": "rejected", "comment": ""}):
            with self.assertRaises(ValueError):
                self.save(**fields)
        self.assertEqual(self.store.history(self.package), [])

    def test_persists_across_instances_and_isolates_packages(self):
        event = self.save(assignee="Служба качества")
        new_store = ReviewStore(self.path)
        self.assertEqual(new_store.history(self.package), [event])
        self.assertFalse(new_store.history("another-package"))
        self.assertTrue(event["timestamp"].endswith("+00:00"))
        self.assertEqual(event["source_ids"], [s.id for s in self.finding.sources])

    def test_cannot_close_without_fix_and_verification(self):
        with self.assertRaises(ValueError):
            self.save(status="closed")
        with self.assertRaises(ValueError):
            self.save(status="corrected")
        event = self.save(status="corrected", correction_reference="Редакция 10, п. 4.2")
        self.assertEqual(gate_status(self.result, [event])["closed"], 0)
        done = self.save(status="closed", expected_revision=1, comment="Исправление сверено с редакцией 10.")
        self.assertEqual(done["correction_reference"], "Редакция 10, п. 4.2")
        self.assertEqual(gate_status(self.result, self.store.history(self.package))["closed"], 1)

    def test_history_preserved_when_reopened(self):
        first = self.save()
        second = self.save(status="found", expected_revision=1, comment="Появились дополнительные данные.")
        events = self.store.history(self.package)
        self.assertEqual(events, [first, second])
        self.assertEqual(current_decisions(events)[finding_id(self.finding)]["status"], "found")

    def test_stale_session_cannot_overwrite_a_decision(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "другой сессии"):
            self.save(expected_revision=0)
        self.assertEqual(len(self.store.history(self.package)), 1)

    def test_corrupted_history_is_not_treated_as_no_decisions(self):
        self.save()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("UPDATE review_events SET event=?", ('{"status":"closed"}',))
            connection.commit()
        with self.assertRaises(ValueError):
            self.store.history(self.package)

    def test_unknown_status_and_out_of_range_priority_rejected(self):
        for fields in ({"status": "nonsense"}, {"impact": 4}, {"impact": True}):
            with self.assertRaises(ValueError):
                self.save(**fields)

    def test_source_free_decision_rejected(self):
        finding = Finding("loss", "Не подтверждено", "Нет источников", .9)
        with self.assertRaises(ValueError):
            self.store.save(self.package, finding, status="intentional", actor="Тест", comment="Тест")

    def test_all_reviewed_green_but_incomplete_or_warning_never_green(self):
        for finding in review_items(self.result):
            self.store.save(self.package, finding, status="rejected", actor="Аудитор", comment="Сверено с другим пунктом.")
        history = self.store.history(self.package)
        self.assertEqual(gate_status(self.result, history)["color"], "green")
        self.result.warnings.append("Неполное извлечение")
        self.assertEqual(gate_status(self.result, history)["color"], "yellow")
        self.result.warnings.clear()
        self.result.analysis_context["after_complete_user_declared"] = False
        self.assertEqual(gate_status(self.result, history)["color"], "yellow")

    def test_manual_priority_three_blocks_hygiene_finding(self):
        result = inspect_document_pack([document_from_lines("new.docx", "after", ["1.1. ;"])], after_complete=True)
        finding = review_items(result)[0]
        event = self.store.save(package_id(result), finding, status="found", actor="Рецензент", comment="Критично для утверждения.", impact=3)
        self.assertEqual(gate_status(result, [event])["color"], "red")
        self.assertEqual(gate_status(result, [event])["blocker_ids"], [finding_id(finding)])

    def test_old_hygiene_does_not_block_new_edition(self):
        before = document_from_lines("old.docx", "before", ["1.1. ;"])
        after = document_from_lines("new.docx", "after", ["1.1. Работа аудиторов."])
        result = inspect_document_pack([after], after_complete=True)
        result.document_checks = lint_documents([before, after])
        self.assertTrue(result.document_checks)
        self.assertFalse(review_items(result))
        self.assertEqual(gate_status(result, [])["color"], "green")

    def test_ids_stable_and_changed_evidence_invalidates_decisions(self):
        before, after = demo_documents()
        same = analyze_documents(before, after, after_complete=True)
        self.assertEqual(self.package, package_id(same))
        self.assertEqual(finding_id(self.finding), finding_id(same.findings[0]))
        same.findings[0].explanation += " Новое основание."
        self.assertNotEqual(finding_id(self.finding), finding_id(same.findings[0]))

    def test_word_and_pdf_contain_decisions_quotes_and_history(self):
        self.save(comment="Сверено <с пунктом> & принято.")
        self.save(status="found", expected_revision=1, comment="Повторная проверка.")
        history = self.store.history(self.package)
        word = WordDocument(BytesIO(approval_docx(self.result, history)))
        word_text = "\n".join(p.text for p in word.paragraphs)
        pdf = PdfReader(BytesIO(approval_pdf(self.result, history)))
        pdf_text = "\n".join(p.extract_text() for p in pdf.pages)
        for text in (word_text, pdf_text):
            text = text.replace("\n", " ")
            self.assertIn("Лист согласования", text)
            self.assertIn("Повторная проверка", text)
            self.assertIn("Сверено <с пунктом> & принято", text)
            self.assertIn("Эксперт", text)
            self.assertIn(self.finding.sources[0].id, text.replace("\n", ""))
            self.assertIn(finding_id(self.finding), text.replace("\n", ""))
            self.assertIn("История решений", text)
            self.assertIn("не электронная подпись", text.replace("\n", " "))
