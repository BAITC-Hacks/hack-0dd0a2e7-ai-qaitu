import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.models import AnalysisResult, Fragment, Function, FunctionMatch, MatrixRow
from qaitu.confidence import assessment
from qaitu.reporting import markdown_report


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def widget(widgets, label):
    return next(item for item in widgets if item.label == label)


class InterfaceTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        return app

    def test_demo_runs_locally_with_matrix_and_explicit_sample_label(self):
        app = self.make_app()
        self.assertFalse(any('qa-result-nav' in item.value for item in app.markdown))
        with patch("qaitu.ai_reviewer.review_with_llm") as reviewer:
            widget(app.button, "Запустить контрольный пример").click().run()
        reviewer.assert_not_called()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(any("КОНТРОЛЬНЫЙ ПРИМЕР" in item.value for item in app.caption))
        self.assertTrue(any("<table" in item.value for item in app.markdown))
        navigation = next(item.value for item in app.markdown if 'qa-result-nav' in item.value)
        destinations = {header.proto.anchor for header in app.header}
        for anchor in ("overview", "conclusion", "structure", "matrix", "sources", "export"):
            self.assertIn(f'href="#{anchor}"', navigation)
            self.assertIn(anchor, destinations)
        self.assertLess(navigation.index('href="#conclusion"'), navigation.index('href="#matrix"'))

    def test_new_welcome_action_runs_the_local_demo(self):
        app = self.make_app()
        welcome_button = next(button for button in app.button if button.key == "welcome_demo")
        welcome_button.click().run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(app.session_state["result"].matrix_rows)
        self.assertEqual(app.session_state["mode"], "demo")

    def test_report_downloads_are_deferred_and_use_unfiltered_result(self):
        import streamlit as st
        download = st.download_button
        app = self.make_app()
        with patch("streamlit.download_button", wraps=download) as buttons, \
             patch("qaitu.pdf_report.pdf_report", return_value=b"%PDF-test") as pdf_export, \
             patch("qaitu.exports.excel_report", return_value=b"xlsx-test") as excel_export:
            widget(app.button, "Запустить контрольный пример").click().run()
            self.assertFalse(app.exception)
            pdf_export.assert_not_called()
            excel_export.assert_not_called()
            calls = {call.args[0]: call for call in buttons.call_args_list}
            self.assertTrue({"Краткий PDF", "Полный PDF", "Excel", "JSON", "Markdown"} <= set(calls))
            self.assertTrue(any("Дополнительно · технический формат" == item.label for item in app.expander))
            short = calls["Краткий PDF"].args[1]
            full = calls["Полный PDF"].args[1]
            xlsx = calls["Excel"].args[1]
            self.assertEqual(short(), b"%PDF-test")
            self.assertFalse(pdf_export.call_args.kwargs["full"])
            full()
            self.assertTrue(pdf_export.call_args.kwargs["full"])
            self.assertEqual(xlsx(), b"xlsx-test")
            result = app.session_state["result"]
            self.assertIs(pdf_export.call_args.args[0], result)
            widget(app.text_input, "Найти функцию или владельца").set_value("нет-такой-функции").run()
            self.assertFalse(app.exception)
            self.assertIs(full.args[0], result)
            self.assertTrue(result.matrix_rows)

    def test_confidence_bands_and_weak_filter_keep_full_export(self):
        app = self.make_app()
        self.assertFalse(widget(app.checkbox, "Все необходимые документы «после» загружены").value)
        widget(app.button, "Запустить контрольный пример").click().run()
        result = app.session_state["result"]
        self.assertTrue(all(f.assessment for f in result.findings))
        self.assertTrue(any("Почему такая оценка" == e.label for e in app.expander))
        finding = result.findings[0]
        finding.title = "Слабый тестовый кандидат"
        finding.confidence = .25
        finding.assessment = assessment(.25, limitations=["Недостаточно данных"])
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(any(finding.title in e.label for e in app.expander))
        self.assertTrue(any("Скрыто слабых сигналов: 1" in c.value for c in app.caption))
        self.assertIn(finding.title, markdown_report(result))
        self.assertIn(finding.title, str(result.to_dict()))
        widget(app.checkbox, "Показать слабые сигналы (менее 40%)").check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(finding.title in e.label and "25%" in e.label for e in app.expander))
        self.assertEqual(len(app.session_state["result"].findings), len(result.findings))

    def test_matrix_pagination_keeps_full_result_and_recovers_after_filtering(self):
        app = self.make_app()
        widget(app.button, "Запустить контрольный пример").click().run()
        result = app.session_state["result"]
        template = result.matrix_rows[0]
        result.matrix_rows = [replace(template, id=f"row-{i}", label=f"Функция {i}") for i in range(23)]
        app.run()
        widget(app.selectbox, "Страница матрицы").set_value(2).run()
        self.assertFalse(app.exception)
        table = next(item.value for item in app.markdown if '<table' in item.value)
        self.assertEqual(table.count('<tr class='), 3)
        widget(app.text_input, "Найти функцию или владельца").set_value("Функция 22").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["result"].matrix_rows), 23)
        table = next(item.value for item in app.markdown if '<table' in item.value)
        self.assertEqual(table.count('<tr class='), 1)

    def test_matrix_escapes_markup_and_filters_keep_full_result(self):
        old = Fragment("old", "old.docx", "Готовит отчёт <script>alert(1)</script>", "п. 1", "before")
        new = Fragment("new", "new.docx", "Готовит отчёт", "п. 2", "after")
        context = Fragment("context", "new.docx", "Директор отдела:", "п. 2, заголовок", "after")
        before = Function("f1", "Отдел <старый>", old.text, old)
        after = Function("f2", "Отдел качества", new.text, new, context_sources=(context,))
        row = MatrixRow("transfer", old.text, [before], [after], "moved")
        result = AnalysisResult([], [FunctionMatch(before, after, .9, "moved")], [], matrix_rows=[row])
        app = self.make_app()
        app.session_state["result"] = result
        app.session_state["mode"] = "uploaded"
        app.run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        table = next(item.value for item in app.markdown if "<table" in item.value)
        self.assertIn("&lt;script&gt;", table)
        self.assertNotIn("<script>", table)
        self.assertIn("+ И", table)
        self.assertTrue(any("Директор отдела:" in item.value for item in app.text))
        app.button(key="expand_matrix").click().run()
        self.assertFalse(app.exception)
        expanded = [item.value for item in app.markdown if "<table" in item.value][-1]
        self.assertIn("&lt;script&gt;", expanded)
        self.assertNotIn("<script>", expanded)
        app.run()
        widget(app.text_input, "Найти функцию или владельца").set_value("несуществующая функция").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("строки не найдены" in item.value for item in app.info))
        self.assertEqual(len(app.session_state["result"].matrix_rows), 1)
        widget(app.text_input, "Найти функцию или владельца").set_value("").run()
        self.assertFalse(app.exception)

    def test_expanded_matrix_uses_filters_and_keeps_report_state(self):
        app = self.make_app()
        widget(app.button, "Запустить контрольный пример").click().run()
        result = app.session_state["result"]
        template = result.matrix_rows[0]
        result.matrix_rows = [replace(template, id=f"row-{i}", label=f"Функция {i}") for i in range(30)]
        app.run()
        widget(app.selectbox, "Страница матрицы").set_value(2).run()
        page_before = widget(app.selectbox, "Страница матрицы").value
        selection_before = app.session_state["matrix_row_selection"]
        with patch("qaitu.analyzer.analyze_documents") as analyze, patch("qaitu.ai_reviewer.review_with_llm") as reviewer:
            app.button(key="expand_matrix").click().run()
        analyze.assert_not_called()
        reviewer.assert_not_called()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        expanded = [item.value for item in app.markdown if "<table" in item.value][-1]
        self.assertEqual(expanded.count('<tr class='), 10)
        self.assertIn("Функция 24", expanded)
        app.run()
        self.assertEqual(widget(app.selectbox, "Страница матрицы").value, page_before)
        self.assertEqual(app.session_state["matrix_row_selection"], selection_before)
        widget(app.text_input, "Найти функцию или владельца").set_value("Функция 29").run()
        app.button(key="expand_matrix").click().run()
        self.assertFalse(app.exception)
        expanded = [item.value for item in app.markdown if "<table" in item.value][-1]
        self.assertEqual(expanded.count('<tr class='), 1)
        self.assertIn("Функция 29", expanded)
        self.assertEqual(len(app.session_state["result"].matrix_rows), 30)

    def test_optional_llm_error_preserves_local_result_without_leaking_error(self):
        app = self.make_app()
        widget(app.checkbox, "Разрешаю отправку фрагментов во внешний API").check().run()
        widget(app.text_input, "OpenAI API key").set_value("not-a-real-key")
        with patch("qaitu.ai_reviewer.review_with_llm", side_effect=RuntimeError("DO NOT LEAK not-a-real-key")) as reviewer:
            widget(app.button, "Запустить контрольный пример").click().run()
        reviewer.assert_called_once()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(app.session_state["result"].matrix_rows)
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("Локальные результаты доступны", warnings)
        self.assertNotIn("not-a-real-key", warnings)
        self.assertNotIn("DO NOT LEAK", warnings)


if __name__ == "__main__":
    unittest.main()
