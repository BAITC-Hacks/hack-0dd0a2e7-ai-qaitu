"""Human decisions, append-only local history, and a conservative review gate.

No decisions change the detector's findings or confidence. The named reviewer is
self-declared, not an authenticated identity or electronic signature.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3

from .models import AnalysisResult, Finding
from .reporting import collect_sources


STATUSES = {
    "found": "Открыто / вернуть на проверку",
    "corrected": "Исправление внесено — ждёт подтверждения",
    "closed": "Исправление проверено — закрыто",
    "intentional": "Осознанное решение — закрыто",
    "rejected": "Ошибка алгоритма — отклонено",
}
CLOSED = {"closed", "intentional", "rejected"}
GATE_LABELS = {"red": "Не готов: есть критические открытые вопросы",
               "yellow": "С замечаниями / недостаточно данных",
               "green": "Чек-лист закрыт"}


def finding_id(finding: Finding) -> str:
    value = [finding.kind, finding.code, finding.title, finding.explanation,
             sorted((s.id, s.text) for s in finding.sources)]
    return "F-" + sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()[:20]


def review_items(result: AnalysisResult) -> list[Finding]:
    # Old-edition hygiene remains visible in the linter but does not block
    # approval of an after edition in which those problems no longer exist.
    candidates = result.findings + [f for f in result.document_checks
                                    if any(s.period == "after" for s in f.sources)]
    return list({finding_id(f): f for f in candidates}.values())


def package_id(result: AnalysisResult) -> str:
    value = sorted((s.id, s.text) for s in collect_sources(result))
    return sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def default_db_path() -> Path:
    return Path(os.environ.get("QAITU_REVIEW_DB", str(Path(__file__).resolve().parents[1] / ".qaitu" / "reviews.sqlite3")))


class ReviewStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_db_path()

    def history(self, package: str) -> list[dict]:
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            if not connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_events'").fetchone():
                return []
            rows = connection.execute("SELECT event FROM review_events WHERE package=? ORDER BY seq", (package,)).fetchall()
        events = [json.loads(row[0]) for row in rows]
        for event in events:
            if not isinstance(event, dict) or not all(key in event for key in (
                    "finding_id", "revision", "status", "actor", "comment", "assignee",
                    "correction_reference", "impact", "timestamp", "source_ids")):
                raise ValueError("Формат журнала решений повреждён или не поддерживается.")
            if (not isinstance(event["status"], str) or event["status"] not in STATUSES or not isinstance(event["revision"], int)
                    or event["revision"] < 1 or event["impact"] not in (1, 2, 3)
                    or not isinstance(event["source_ids"], list)
                    or not all(isinstance(event[key], str) for key in (
                        "finding_id", "actor", "comment", "assignee", "correction_reference", "timestamp"))):
                raise ValueError("Некорректные поля журнала решений.")
        return events

    def save(self, package: str, finding: Finding, *, status: str, actor: str,
             comment: str, assignee: str = "", correction_reference: str = "",
             impact: int = 2, expected_revision: int = 0) -> dict:
        if status not in STATUSES:
            raise ValueError("Неизвестный статус решения.")
        if not actor.strip() or not comment.strip():
            raise ValueError("Укажите имя эксперта и содержательный комментарий.")
        if not finding.sources or any(not s.id or not s.text.strip() for s in finding.sources):
            raise ValueError("Нельзя согласовать вывод без источника и цитаты.")
        if not isinstance(impact, int) or isinstance(impact, bool) or impact not in (1, 2, 3):
            raise ValueError("Экспертный приоритет должен быть 1, 2 или 3.")
        if any(len(value) > limit for value, limit in ((actor, 200), (assignee, 200), (comment, 5000), (correction_reference, 1000))):
            raise ValueError("Текст решения превышает допустимую длину.")
        if status == "corrected" and not correction_reference.strip():
            raise ValueError("Укажите, где внесено исправление: файл/редакция и пункт.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS review_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, package TEXT NOT NULL, finding TEXT NOT NULL, revision INTEGER NOT NULL, event TEXT NOT NULL, UNIQUE(package, finding, revision))")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT event FROM review_events WHERE package=? AND finding=? ORDER BY revision DESC LIMIT 1",
                                     (package, finding_id(finding))).fetchone()
            previous = json.loads(row[0]) if row else {}
            if previous.get("revision", 0) != expected_revision:
                raise ValueError("Решение уже изменено в другой сессии. Обновите страницу и повторите проверку.")
            if status == "closed" and previous.get("status") != "corrected":
                raise ValueError("Сначала зафиксируйте внесённое исправление, затем подтвердите его проверку.")
            if status == "closed":
                correction_reference = previous["correction_reference"]
            event = {
                "finding_id": finding_id(finding), "revision": expected_revision + 1,
                "status": status, "actor": actor.strip(), "assignee": assignee.strip(),
                "comment": comment.strip(), "correction_reference": correction_reference.strip(),
                "impact": impact, "impact_method": "human-priority-not-risk-matrix",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ids": [s.id for s in finding.sources],
            }
            connection.execute("INSERT INTO review_events(package, finding, revision, event) VALUES(?,?,?,?)",
                               (package, finding_id(finding), event["revision"], json.dumps(event, ensure_ascii=False)))
            connection.commit()
        return event


def current_decisions(history: list[dict]) -> dict[str, dict]:
    return {event["finding_id"]: event for event in history}


def gate_status(result: AnalysisResult, history: list[dict]) -> dict:
    decisions = current_decisions(history)
    items = review_items(result)
    pending = [f for f in items if decisions.get(finding_id(f), {}).get("status", "found") not in CLOSED]
    blockers = [f for f in pending if f.kind in {"loss", "conflict"} or
                decisions.get(finding_id(f), {}).get("impact", 2) == 3]
    reasons = []
    if blockers:
        color = "red"
        reasons.append("Открыты потери, конфликты ролей или вопросы с экспертным приоритетом 3.")
    elif pending:
        color = "yellow"
        reasons.append("Остаются вопросы для проверки или исправления без подтверждения эксперта.")
    else:
        color = "green"
    if not result.analysis_context.get("after_complete_user_declared", False):
        reasons.append("Полнота комплекта «после» пользователем не подтверждена.")
        if color == "green":
            color = "yellow"
    if result.warnings or not any(s.period == "after" for s in collect_sources(result)):
        reasons.append("Есть ограничения обработки или нет текста новой редакции.")
        if color == "green":
            color = "yellow"
    return {"color": color, "label": GATE_LABELS[color], "open": len(pending),
            "closed": len(items) - len(pending), "total": len(items),
            "blocker_ids": [finding_id(f) for f in blockers], "reasons": reasons,
            "limitation": "Статус только по найденным вопросам, не разрешение на утверждение и не юридическое заключение. Матрица ущерба F5 пока не рассчитывается; приоритет 1–3 задаёт эксперт."}
