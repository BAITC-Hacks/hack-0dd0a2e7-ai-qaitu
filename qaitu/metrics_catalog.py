"""Extensible metric dictionary; ambiguous labels never resolve by proximity."""
from __future__ import annotations

import re
from .metrics_models import MetricDefinition


def _definition(code, name, unit, better, aliases, aggregation="mean", normalize_by=None):
    return MetricDefinition(code, name, unit, better, [code, name, *aliases], normalize_by, aggregation)


METRIC_CATALOG = {item.code: item for item in [
    _definition("audits_planned", "Проверок по плану", "шт.", "none", ["План проверок", "Запланировано проверок", "Плановое количество проверок", "Количество проверок"], "sum", "fte_actual"),
    _definition("audits_done", "Проверок проведено", "шт.", "up", ["Проведено проверок", "Проведено аудитов", "Выполнено проверок", "Количество проведенных проверок", "Количество проверок"], "sum", "fte_actual"),
    _definition("plan_completion", "Выполнение плана работ", "%", "up", ["Выполнение плана", "Процент выполнения плана", "Выполнение плана работ, %"]),
    _definition("findings_total", "Выявлено нарушений/недостатков", "шт.", "context", ["Выявлено нарушений", "Выявлено недостатков", "Количество нарушений", "Всего нарушений"], "sum", "fte_actual"),
    _definition("findings_material", "Существенных нарушений", "шт.", "context", ["Существенные нарушения", "Выявлено существенных нарушений"], "sum", "fte_actual"),
    _definition("recs_issued", "Выдано рекомендаций", "шт.", "none", ["Количество рекомендаций", "Рекомендаций выдано"], "sum", "fte_actual"),
    _definition("recs_accepted", "Принято рекомендаций", "%", "up", ["Принято рекомендаций, %", "Доля принятых рекомендаций", "Процент принятых рекомендаций"]),
    _definition("recs_implemented", "Выполнено мероприятий в срок", "%", "up", ["Выполнено мероприятий в срок, %", "Доля мероприятий в срок", "Рекомендации выполнены в срок"]),
    _definition("action_overdue", "Просроченных мероприятий", "шт.", "down", ["Просроченные мероприятия", "Количество просроченных мероприятий"], "last"),
    _definition("audit_cycle_days", "Длительность аудиторского цикла", "дни", "down", ["Длительность аудиторского цикла, дней", "Срок проверки", "Продолжительность аудита", "Средняя длительность цикла"]),
    _definition("report_delay_days", "Задержка сдачи отчётов СД/КпА", "дни", "down", ["Задержка отчетов", "Задержка сдачи отчетов", "Задержка сдачи отчетов СД/КпА, дней", "Задержка отчетности", "Задержка отчета в днях"]),
    _definition("headcount", "Штатная численность", "чел.", "none", ["Численность по штату", "Штат", "Штатная численность персонала"], "last"),
    _definition("fte_actual", "Фактическая численность", "FTE", "none", ["Фактическая численность персонала", "Фактическая численность FTE", "Фактические FTE", "FTE"], "last"),
    _definition("budget_plan", "Бюджет затрат БВА — план", "ден. ед.", "none", ["Бюджет план", "План бюджета", "План затрат", "Плановый бюджет"], "sum"),
    _definition("budget_fact", "Бюджет затрат БВА — факт", "ден. ед.", "none", ["Бюджет факт", "Фактический бюджет", "Факт затрат", "Фактические затраты"], "sum"),
    _definition("assurance_coverage", "Покрытие рисков Картой гарантий", "%", "up", ["Покрытие рисков Картой гарантий, %", "Покрытие рисков", "Доля покрытия рисков"]),
    _definition("qa_score", "Оценка качества", "баллы", "up", ["Оценка качества внутренняя", "Оценка качества внешняя", "Внутренняя оценка качества", "Внешняя оценка качества"]),
    _definition("hotline_cases", "Обращения на горячую линию", "шт.", "none", ["Обращений на горячую линию", "Поступило обращений на горячую линию"], "sum", "fte_actual"),
    _definition("hotline_closed", "Закрыто обращений на горячую линию", "шт.", "up", ["Закрытые обращения", "Обращений закрыто", "Горячая линия закрыто"], "sum", "fte_actual"),
]}


def normalize_label(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"\((?:%|шт\.?|ед\.?|дн\.?|дней|чел\.?|fte|балл\w*)\)", " ", text)
    text = re.sub(r"[,;]\s*(?:%|шт\.?|ед\.?|дн\.?|дней|чел\.?|fte|балл\w*)\s*$", "", text)
    return " ".join(re.findall(r"[а-яa-z0-9]+", text))


def resolve_metric(label: object) -> list[str]:
    normalized = normalize_label(label)
    return [code for code, definition in METRIC_CATALOG.items()
            if any(normalize_label(alias) == normalized for alias in definition.aliases)] if normalized else []
