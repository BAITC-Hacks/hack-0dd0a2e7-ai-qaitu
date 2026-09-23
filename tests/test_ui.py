import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.config import OpenAISettings
from qaitu.models import AnalysisResult, Fragment, Function, FunctionMatch, MatrixRow
from qaitu.confidence import assessment
from qaitu.reporting import markdown_report


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def widget(widgets, label):
    return next(item for item in widgets if item.label == label and item.key != "welcome_demo")


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        # Never read a developer's configured key or call a paid service in tests.
        self.settings_patch = patch("qaitu.config.load_openai_settings", return_value=OpenAISettings())
        self.settings = self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.agent_patch = patch("qaitu.ai_agent.run_comparison_agent")
        self.agent = self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)

    def make_app(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        return app

    def test_demo_runs_locally_with_matrix_and_explicit_sample_label(self):
        app = self.make_app()
        self.assertFalse(any('qa-result-nav' in item.value for item in app.markdown))
        widget(app.button, "Запустить контрольный пример").click().run()
        self.agent.assert_not_called()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(any("КОНТРОЛЬНЫЙ ПРИМЕР" in item.value for item in app.caption))
        self.assertTrue(any("<table" in item.value for item in app.markdown))
        navigation = next(item.value for item in app.markdown if 'qa-result-nav' in item.value)
        destinations = {header.proto.anchor for header in app.header}
        for anchor in ("overview", "conclusion", "semantic", "structure", "matrix", "sources", "export"):
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

    def test_agent_error_preserves_local_result_without_leaking_error(self):
        app = self.make_app()
        widget(app.radio, "Режим сравнения").set_value("ИИ + локальное сравнение").run()
        widget(app.text_input, "OpenAI API key — заменить для сеанса").set_value("not-a-real-key")
        self.agent.side_effect = RuntimeError("DO NOT LEAK not-a-real-key")
        widget(app.button, "Запустить контрольный пример").click().run()
        self.agent.assert_called_once()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertTrue(app.session_state["result"].matrix_rows)
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("Локальные результаты доступны", warnings)
        self.assertNotIn("not-a-real-key", warnings)
        self.assertNotIn("DO NOT LEAK", warnings)

    def test_saved_key_runs_only_on_explicit_action_and_is_never_prefilled(self):
        saved_key = "private-saved-test-secret"
        self.settings.return_value = OpenAISettings(api_key=saved_key, model="test-model")

        def reviewed(before_docs, after_docs, result, api_key, model, progress=None):
            self.assertEqual(api_key, saved_key)
            self.assertEqual(model, "test-model")
            before_source = before_docs[0].fragments[0]
            after_source = after_docs[0].fragments[0]
            if progress:
                progress(1, 1, "Пакет проверен")
            result.ai_review = {"status": "completed", "model": model, "summary": "Найдено одно проверяемое соответствие.", "completed_batches": 1, "total_batches": 1,
                "covered_sources": 2, "total_sources": 2, "input_tokens": 100, "output_tokens": 20, "comparisons": [{"kind": "retained", "title": "Независимая оценка сохранена",
                    "explanation": "Смысл обязанности сохранён.", "before_source_ids": [before_source.id], "after_source_ids": [after_source.id],
                    "evidence": [{"source_id": before_source.id, "quote": before_source.text}, {"source_id": after_source.id, "quote": after_source.text}], "recommendation": "Проверить область действия."}]}

        self.agent.side_effect = reviewed
        app = self.make_app()
        self.agent.assert_not_called()
        self.assertEqual(widget(app.radio, "Режим сравнения").value, "ИИ + локальное сравнение")
        self.assertEqual(widget(app.text_input, "OpenAI API key — заменить для сеанса").value, "")
        widget(app.button, "Запустить контрольный пример").click().run()
        self.agent.assert_called_once()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(app.session_state["result"].ai_review["status"], "completed")
        self.assertTrue(any("Независимая оценка сохранена" in item.label for item in app.expander))
        self.assertTrue(any("осуществляет независимую оценку" in item.value for item in app.text))
        self.assertNotIn(saved_key, str(app.session_state["result"].to_dict()))
        widget(app.text_input, "Найти функцию или владельца").set_value("аудит").run()
        app.run()
        self.agent.assert_called_once()
        rendered = "\n".join(item.value for item in [*app.markdown, *app.text, *app.caption, *app.warning])
        self.assertNotIn(saved_key, rendered)

    def test_partial_result_is_labeled_and_keeps_local_matrix(self):
        self.settings.return_value = OpenAISettings(api_key="fake-server-key")

        def partial(before_docs, after_docs, result, api_key, model, progress=None):
            result.ai_review = {"status": "partial", "model": model, "summary": "Обработан один пакет.", "comparisons": [], "covered_sources": 4, "total_sources": 10, "completed_batches": 1, "total_batches": 2, "error": "Второй пакет недоступен."}

        self.agent.side_effect = partial
        app = self.make_app()
        widget(app.button, "Запустить контрольный пример").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["result"].matrix_rows)
        self.assertTrue(any("Частичный результат" in item.value for item in app.warning))
        self.assertTrue(any(item.value == "4 / 10" for item in app.metric))
        self.assertFalse(any("ИИ-сравнение завершено" in item.value for item in app.success))

    def test_local_mode_with_saved_key_still_makes_no_api_call(self):
        self.settings.return_value = OpenAISettings(api_key="fake-server-key")
        app = self.make_app()
        widget(app.radio, "Режим сравнения").set_value("Локальное сравнение").run()
        widget(app.button, "Запустить контрольный пример").click().run()
        self.agent.assert_not_called()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["result"].ai_review["status"], "skipped")

    def test_welcome_demo_is_local_even_with_saved_key(self):
        self.settings.return_value = OpenAISettings(api_key="fake-server-key")
        app = self.make_app()
        next(item for item in app.button if item.key == "welcome_demo").click().run()
        self.agent.assert_not_called()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["result"].matrix_rows)


if __name__ == "__main__":
    unittest.main()
