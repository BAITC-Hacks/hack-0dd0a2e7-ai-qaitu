"""Read server-side OpenAI settings without exposing credentials to widgets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    config_error: str = ""


def load_openai_settings(
    secrets_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> OpenAISettings:
    """Environment takes precedence over an ignored project secrets file.

    Reading configuration never contacts the API or writes secrets. Avoid
    returning parser exception text: it can contain the malformed secret line.
    """
    env = os.environ if environ is None else environ
    path = secrets_path or Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    config: dict = {}
    error = ""
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        error = "Не удалось прочитать локальные настройки OpenAI. Проверьте формат secrets.toml или переменные окружения."

    def setting(name: str, default: str = "") -> str:
        value = env.get(name) or config.get(name, default)
        return value.strip() if isinstance(value, str) else default

    return OpenAISettings(
        api_key=setting("OPENAI_API_KEY"),
        model=setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        config_error=error,
    )
