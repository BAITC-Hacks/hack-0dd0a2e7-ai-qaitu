import tempfile
import unittest
from pathlib import Path

from qaitu.config import DEFAULT_MODEL, load_openai_settings


class OpenAIConfigTests(unittest.TestCase):
    def test_missing_settings_keep_local_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = load_openai_settings(Path(folder) / "missing.toml", {})
        self.assertEqual(settings.api_key, "")
        self.assertEqual(settings.model, DEFAULT_MODEL)

    def test_server_secret_is_read_without_appearing_in_repr(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "secrets.toml"
            path.write_text('OPENAI_API_KEY = "private-test-value"\nOPENAI_MODEL = "test-model"\n')
            settings = load_openai_settings(path, {})
        self.assertEqual(settings.api_key, "private-test-value")
        self.assertEqual(settings.model, "test-model")
        self.assertNotIn(settings.api_key, repr(settings))

    def test_environment_overrides_secret_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "secrets.toml"
            path.write_text('OPENAI_API_KEY = "file-value"\n')
            settings = load_openai_settings(path, {"OPENAI_API_KEY": "env-value", "OPENAI_MODEL": "env-model"})
        self.assertEqual(settings.api_key, "env-value")
        self.assertEqual(settings.model, "env-model")

    def test_invalid_file_does_not_echo_secret_in_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "secrets.toml"
            path.write_text('OPENAI_API_KEY = "private-test-value')
            settings = load_openai_settings(path, {})
        self.assertTrue(settings.config_error)
        self.assertNotIn("private-test-value", settings.config_error)
        self.assertEqual(settings.api_key, "")


if __name__ == "__main__":
    unittest.main()
