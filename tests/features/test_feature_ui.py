from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from qaitu.extractors import document_from_lines
from qaitu.linter import inspect_document_pack
from qaitu.review import ReviewStore, package_id


APP = str(Path(__file__).resolve().parents[2] / "app.py")


def widget(widgets, label):
    return next(w for w in widgets if w.label == label)


class FeatureInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "reviews.sqlite3"
        self.env = patch.dict("os.environ", {"QAITU_REVIEW_DB": str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def app(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def demo(self):
        app = self.app()
        widget(app.button, "Запустить контрольный пример").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        return app

    def test_switch_modes_without_reanalysis_or_api(self):
        app = self.demo()
        result = app.session_state["result"]
        with patch("qaitu.analyzer.analyze_documents") as analyze, patch("qaitu.ai_reviewer.review_with_llm") as llm:
            for mode in ("Проверка проекта", "Пакет для СД / Комитета", "Разбор реорганизации"):
                widget(app.selectbox, "Режим работы").set_value(mode).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertIs(app.session_state["result"], result)
            analyze.assert_not_called()
            llm.assert_not_called()

    def test_single_edition_view_does_not_claim_new_units(self):
        app = self.app()
        doc = document_from_lines("draft.docx", "after", ["1.1. ;"])
        app.session_state["result"] = inspect_document_pack([doc])
        app.session_state["mode"] = "uploaded"
        app.run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any("Проверена одна редакция" in i.value for i in app.info))
        self.assertTrue(any("LNT-EMPTY" in e.label for e in app.expander))
        self.assertFalse(any(h.value == "Матрица функций" for h in app.header))
        widget(app.selectbox, "Режим работы").set_value("Проверка проекта").run()
        self.assertTrue(any(b.label == "Проверить проект" for b in app.button))

    def test_blank_comment_cannot_save_then_valid_decision_persists(self):
        app = self.demo()
        widget(app.text_input, "Кто принимает решение").set_value("Рецензент")
        widget(app.selectbox, "Решение эксперта").set_value("intentional")
        widget(app.button, "Сохранить решение").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("содержательный комментарий" in e.value for e in app.error))
        package = package_id(app.session_state["result"])
        self.assertEqual(ReviewStore(self.path).history(package), [])
        widget(app.text_area, "Обоснование решения (обязательно)").set_value("Ответственность закреплена в отдельном регламенте, решение согласовано.")
        widget(app.button, "Сохранить решение").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        history = ReviewStore(self.path).history(package)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "intentional")
        fresh = self.demo()
        self.assertTrue(any("Осознанное решение — закрыто" in c.value for c in fresh.caption))
        self.assertTrue(any("История решений · 1" == e.label for e in fresh.expander))

    def test_read_only_render_does_not_write_review_database(self):
        self.demo()
        self.assertFalse(self.path.exists())

    def test_export_callbacks_deferred_and_snapshot_not_filtered(self):
        import streamlit as st
        real_download = st.download_button
        with patch("streamlit.download_button", wraps=real_download) as buttons, \
             patch("qaitu.approval_export.approval_docx", return_value=b"docx-test") as word, \
             patch("qaitu.approval_export.approval_pdf", return_value=b"pdf-test") as pdf:
            # review_ui may already be imported; patch the names it actually calls.
            with patch("qaitu.review_ui.approval_docx", word), patch("qaitu.review_ui.approval_pdf", pdf):
                app = self.demo()
            word.assert_not_called()
            pdf.assert_not_called()
            calls = {c.args[0]: c for c in buttons.call_args_list}
            self.assertEqual(calls["Лист согласования · Word"].args[1](), b"docx-test")
            self.assertEqual(calls["Лист согласования · PDF"].args[1](), b"pdf-test")
            self.assertIs(word.call_args.args[0], app.session_state["result"])

    def test_unreadable_history_does_not_show_green_or_fake_empty_state(self):
        with patch("qaitu.review.ReviewStore.history", side_effect=OSError("unavailable")):
            app = self.demo()
        self.assertFalse(app.exception)
        self.assertTrue(any("История решений недоступна" in e.value for e in app.error))
        self.assertTrue(any("Статус согласования неизвестен" in w.value for w in app.warning))
        self.assertFalse(any("Чек-лист закрыт" in s.value for s in app.success))
