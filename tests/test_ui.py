import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.models import AnalysisResult, Fragment, Function, FunctionMatch, MatrixRow


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
