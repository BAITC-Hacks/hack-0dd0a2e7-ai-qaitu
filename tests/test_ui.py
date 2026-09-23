import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.config import OpenAISettings
from qaitu.models import AnalysisResult, Fragment, Function, FunctionMatch, MatrixRow


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
        self.assertTrue(any("КОНТРОЛЬНЫЙ ПРИМЕР" in item.value for item in app.info))
        self.assertTrue(any("<table" in item.value for item in app.markdown))
        navigation = next(item.value for item in app.markdown if 'qa-result-nav' in item.value)
        destinations = {header.proto.anchor for header in app.header}
        for anchor in ("overview", "matrix", "conclusion", "structure", "sources", "export"):
            self.assertIn(f'href="#{anchor}"', navigation)
            self.assertIn(anchor, destinations)

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
        widget(app.text_input, "Найти функцию или владельца").set_value("несуществующая функция").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("строки не найдены" in item.value for item in app.info))
        self.assertEqual(len(app.session_state["result"].matrix_rows), 1)
        widget(app.text_input, "Найти функцию или владельца").set_value("").run()
        self.assertFalse(app.exception)

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


if __name__ == "__main__":
    unittest.main()
