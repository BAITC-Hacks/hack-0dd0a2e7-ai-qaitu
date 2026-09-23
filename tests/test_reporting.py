import unittest

from qaitu.models import AnalysisResult, Fragment, Function, FunctionMatch, MatrixRow
from qaitu.reporting import collect_sources, compact_unit_labels, markdown_report, ordered_matrix_rows


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.old = Fragment("old:1", "old.docx", "Готовит отчёт <script>alert(1)</script>", "п. 1.2", "before")
        self.new = Fragment("new:1", "new.docx", "Готовит отчёт", "п. 5.3", "after")
        self.context = Fragment("new:header", "new.docx", "Директор отдела качества:", "п. 5", "after")
        before = Function("f1", "Отдел отчётности", self.old.text, self.old)
        after = Function("f2", "Отдел качества", self.new.text, self.new, context_sources=(self.context,))
        self.row = MatrixRow("r1", self.old.text, [before], [after], "moved")
        self.result = AnalysisResult([], [FunctionMatch(before, after, .8, "moved")], [], matrix_rows=[self.row])

    def test_catalog_keeps_both_sides_and_ownership_context(self):
        self.assertEqual({source.id for source in collect_sources(self.result)}, {"old:1", "new:1", "new:header"})

    def test_report_contains_full_citations_and_escapes_document_markup(self):
        report = markdown_report(self.result)
        for value in ("old.docx", "new.docx", "п. 1.2", "п. 5.3", "new:header", "Отдел качества"):
            self.assertIn(value, report)
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn("<script>", report)
        self.assertIn("Режим: синтетический контрольный пример", markdown_report(self.result, is_demo=True))

    def test_review_order_prioritizes_changes_and_departments_without_mutation(self):
        global_function = Function("global", "БВА (общие функции)", "Готовит отчёт", self.old)
        preserved = MatrixRow("keep", "Keep", self.row.before, self.row.after, "preserved")
        global_row = MatrixRow("global", "Global", [global_function], [], "moved")
        rows = [preserved, global_row, self.row]
        self.assertEqual([row.id for row in ordered_matrix_rows(rows)], ["r1", "global", "keep"])
        self.assertEqual([row.id for row in rows], ["keep", "global", "r1"])
        self.assertEqual(global_row.status, "moved")

    def test_compact_headers_remain_unique_and_have_full_name_mapping(self):
        units = ["П2", "Отдел с очень длинным названием", "Главный аудитор", "Департамент операционного аудита"]
        labels = compact_unit_labels(units)
        self.assertEqual(len(set(labels.values())), len(units))
        self.assertEqual(set(labels), set(units))
        self.assertEqual(labels["Департамент операционного аудита"], "ДОА")


if __name__ == "__main__":
    unittest.main()
