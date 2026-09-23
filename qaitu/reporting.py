"""Readable, source-complete exports; no model calls or document side effects."""
from __future__ import annotations

from collections import Counter

from .models import AnalysisResult, Fragment


STATUS_LABELS = {
    "preserved": "Сохранено", "created": "Появилось в перечне", "removed": "Не найдено в новом перечне",
    "transformed": "Возможное преобразование", "moved": "Передано / изменены владельцы",
    "changed": "Изменена формулировка", "lost": "Закрепление не найдено",
    "new": "Новое закрепление", "unknown": "Недостаточно данных",
}
NORM_LABELS = {"duty": "Обязанность", "right": "Право", "prohibition": "Запрет"}
ROLE_LABELS = {
    "execute": "исполнение", "participate": "участие", "approve": "утверждение",
    "agree": "согласование", "coordinate": "координация", "control": "контроль",
    "audit": "контроль", "review": "проверка", "unknown": "роль не определена",
}
FINDING_LABELS = {"loss": "Возможная потеря", "duplicate": "Возможное дублирование", "conflict": "Потенциальный конфликт"}
AI_STATUS_LABELS = {"completed": "Завершено", "partial": "Частично", "failed": "Не завершено", "skipped": "Не запускалось", "not_needed": "Нет кандидатов для проверки", "running": "Выполняется", "budget_exhausted": "Истёк лимит времени"}
AI_COMPARISON_LABELS = {"retained": "Сохранено по смыслу", "changed": "Изменено", "moved": "Передано", "added": "Добавлено", "possibly_lost": "Возможная потеря"}
COVERAGE_LABELS = {
    "documents_before": "Документов до", "documents_after": "Документов после",
    "fragments_before": "Фрагментов до", "fragments_after": "Фрагментов после",
    "functions_before": "Назначений до", "functions_after": "Назначений после",
    "unknown_owners_before": "Неустановленных владельцев до",
    "unknown_owners_after": "Неустановленных владельцев после", "matrix_rows": "Строк матрицы",
}


def ordered_matrix_rows(rows):
    """Order for review without changing analysis statuses or ownership."""
    priorities = {"moved": 0, "changed": 1, "lost": 2, "new": 3, "unknown": 4, "preserved": 5}

    def global_only(row):
        return all(function.unit.startswith(("Главный аудитор", "БВА", "Роль:", "Владелец не установлен")) for function in row.before + row.after)

    return sorted(rows, key=lambda row: (priorities.get(row.status, 6), global_only(row)))


def compact_unit_labels(units: list[str]) -> dict[str, str]:
    known = {
        "департамент ит-аудита и анализа данных": "ДИТААД",
        "департамент операционного аудита": "ДОА",
        "департамент непрерывного мониторинга системы внутреннего контроля": "ДНМ",
        "департамент контроля качества аудита и методологии": "ДККМ",
        "бва (общие функции)": "БВА", "главный аудитор": "ГА", "владелец не установлен": "?",
    }
    labels = {}
    used = set()
    for index, unit in enumerate(units, 1):
        candidate = known.get(unit.lower(), unit if len(unit) <= 8 else f"П{index}")
        if candidate in used:
            candidate = f"П{index}"
        while candidate in used:
            candidate += "′"
        labels[unit] = candidate
        used.add(candidate)
    return labels


def collect_sources(result: AnalysisResult) -> list[Fragment]:
    """Include both function sides and inherited ownership/modality context."""
    sources: dict[str, Fragment] = {source.id: source for source in result.sources}
    for item in [*result.unit_changes, *result.findings]:
        for source in item.sources:
            sources[source.id] = source
    functions = []
    for match in result.function_matches:
        functions.extend(function for function in (match.before, match.after) if function)
    for row in result.matrix_rows:
        functions.extend([*row.before, *row.after])
    for function in functions:
        for source in (function.source, *function.context_sources):
            sources[source.id] = source
    return list(sources.values())


def _md(value: str) -> str:
    value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for char in ("\\", "`", "*", "_", "[", "]", "#", "|", "!"):
        value = value.replace(char, "\\" + char)
    return value


def markdown_report(result: AnalysisResult, *, is_demo: bool = False) -> str:
    lines = ["# QAITU · Анализ организационных изменений", "",
             "Режим: синтетический контрольный пример." if is_demo else "Режим: анализ загруженных документов.", "",
             "Выводы — кандидаты для экспертной проверки. Отсутствие назначения в обработанном тексте не доказывает прекращение работы. Несколько владельцев не доказывают дублирование без сопоставления ролей и области ответственности.", "",
             "## Охват", ""]
    sources = collect_sources(result)
    documents = {(source.period, source.document) for source in sources}
    counts = Counter(finding.kind for finding in result.findings)
    units = Counter(change.status for change in result.unit_changes)
    assignments = Counter(row.status for row in result.matrix_rows)
    lines.extend([f"- Документов: {len(documents)}; фрагментов в каталоге: {len(sources)}.",
                  f"- Строк матрицы: {len(result.matrix_rows)}.",
                  f"- Подразделения: сохранено {units['preserved']}, возможно преобразовано {units['transformed']}, появилось в перечне {units['created']}, не найдено в новом перечне {units['removed']}.",
                  f"- Назначения функций: сохранено {assignments['preserved']}, передано или изменены владельцы {assignments['moved']}, изменена формулировка {assignments['changed']}, без прямого соответствия до {assignments['new']}.",
                  f"- Возможных потерь: {counts['loss']}; пересечений: {counts['duplicate']}; конфликтов ролей: {counts['conflict']}.", ""])
    if result.warnings:
        lines.extend(["### Ограничения обработки", ""])
        lines.extend(f"- {_md(warning)}" for warning in result.warnings)
        lines.append("")
    if result.coverage:
        lines.extend(["Показатели извлечения (это не оценка точности):", ""])
        lines.extend(f"- {_md(COVERAGE_LABELS.get(key, key))}: {value}" for key, value in result.coverage.items())
        lines.append("")
    review = getattr(result, "ai_review", {}) or {}
    lines.extend(["## Смысловое ИИ-сравнение", "",
                  f"Статус: {_md(AI_STATUS_LABELS.get(review.get('status', 'skipped'), review.get('status', 'skipped')))}.", "",
                  "ИИ-выводы — отдельный слой анализа. Матрица назначений рассчитана локальным алгоритмом и не переписана моделью.", ""])
    if review.get("model"):
        lines.extend([f"Модель: {_md(review['model'])}.", ""])
    if review.get("summary"):
        lines.extend([_md(review["summary"]), ""])
    if review.get("error"):
        lines.extend([f"Ограничение: {_md(review['error'])}", ""])
    if review.get("verification_status"):
        verification = AI_STATUS_LABELS.get(review["verification_status"], review["verification_status"])
        lines.extend([f"Дополнительная ИИ-проверка обоснованности: {_md(verification)}. Отклонено: {review.get('verification_rejected', 0)}.", ""])
    if review and review.get("status") != "skipped":
        lines.extend([f"Фрагментов в пакетах с проверенным ответом: {review.get('covered_sources', 0)} / {review.get('total_sources', len(sources))}.",
                      f"Фрагментов передано в запросах: {review.get('sent_sources', review.get('covered_sources', 0))}.",
                      f"Успешных пакетов: {review.get('completed_batches', 0)} / {review.get('total_batches', 0)}.",
                      f"Запросов: {review.get('requests_made', 0)}.",
                      f"Не прошли проверку и исключены из выводов: {review.get('rejected_items', 0)}.", "",
                      "Охват фрагментов не означает полноту найденных изменений. Цитаты и идентификаторы проверены автоматически; смысл выводов требует проверки сотрудником.", ""])
        if review.get("usage_available"):
            lines.extend([f"API сообщил входных токенов: {review.get('input_tokens', 0)}; выходных токенов: {review.get('output_tokens', 0)}.", ""])
            if not review.get("usage_complete", False):
                lines.extend(["Данные о расходе неполные: использование получено не для всех запросов. Указанные суммы относятся только к полученным данным.", ""])
        else:
            lines.extend(["Данные о токенах недоступны: API не вернул расход.", ""])
    for number, comparison in enumerate(review.get("comparisons", []), 1):
        lines.extend([f"### ИИ {number}. {_md(AI_COMPARISON_LABELS.get(comparison.get('kind'), 'Сопоставление'))}: {_md(comparison.get('title', ''))}", "",
                      _md(comparison.get("explanation", "")), "", f"Рекомендация: {_md(comparison.get('recommendation', ''))}", ""])
        for period, label in (("before", "До"), ("after", "После")):
            ids = comparison.get(period + "_source_ids", [])
            lines.extend([label + ": " + (", ".join(_md(source_id) for source_id in ids) or "прямое соответствие не приведено"), ""])
        for item in comparison.get("evidence", []):
            lines.extend([f"Источник: {_md(item.get('source_id', ''))}", ""])
            lines.extend("> " + _md(line) for line in item.get("quote", "").splitlines())
            lines.append("")
    lines.extend(["## Аналитическое заключение", ""])
    if not result.findings:
        lines.extend(["Индикаторы рисков не найдены. Это не подтверждение отсутствия рисков.", ""])
    for number, finding in enumerate(result.findings, 1):
        lines.extend([f"### {number}. {_md(FINDING_LABELS.get(finding.kind, finding.kind))}: {_md(finding.title)}", "",
                      _md(finding.explanation), "", f"Рекомендация: {_md(finding.recommendation)}", "",
                      "Источники: " + ", ".join(_md(source.id) for source in finding.sources), ""])
    lines.extend(["## Изменения структуры", ""])
    for change in result.unit_changes:
        lines.append(f"- **{_md(STATUS_LABELS.get(change.status, change.status))}**: {_md(change.before or '—')} → {_md(change.after or '—')}. Источники: " + ", ".join(_md(source.id) for source in change.sources))
    lines.extend(["", "## Функции и назначения", ""])
    for number, row in enumerate(result.matrix_rows, 1):
        lines.extend([f"### {number}. {_md(row.label)}", "",
                      f"{_md(NORM_LABELS.get(row.norm_type, row.norm_type))} · {_md(STATUS_LABELS.get(row.status, row.status))}", ""])
        if row.candidate_overlap:
            lines.extend(["Проверить пересечение областей ответственности нескольких владельцев.", ""])
        for note in row.notes:
            lines.extend([_md(note), ""])
        for label, assignments in (("До", row.before), ("После", row.after)):
            lines.extend([f"**{label}**", ""])
            if not assignments:
                lines.append("Закрепление в обработанных документах не найдено.")
            for function in assignments:
                lines.extend([f"- {_md(function.unit)} · {_md(ROLE_LABELS.get(function.role, function.role))}" + (" · владелец требует проверки" if not function.owner_known else ""),
                              f"  {_md(function.text)}", f"  Источник: {_md(function.source.id)}."])
                if function.scope:
                    lines.append(f"  Область: {_md(function.scope)}.")
                if function.context_sources:
                    lines.append("  Контекст: " + ", ".join(_md(source.id) for source in function.context_sources) + ".")
            lines.append("")
    lines.extend(["## Полный каталог источников", ""])
    for source in sources:
        lines.extend([f"### {_md(source.id)}", "",
                      f"{'До' if source.period == 'before' else 'После'} · {_md(source.document)} · {_md(source.locator)}", ""])
        lines.extend("> " + _md(line) for line in source.text.splitlines())
        lines.append("")
    return "\n".join(lines)
