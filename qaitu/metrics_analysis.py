"""Deterministic, source-backed metrics comparisons, not causal estimates.

The optional module preserves a distinction between arithmetic, a proposed
mechanism, and a confirmed causal effect (which this prototype never claims).
"""
from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import math
import statistics

from .metrics_catalog import METRIC_CATALOG
from .metrics_models import (Attribution, Hypothesis, KnowledgeRecord, MetricValue,
                             PeriodComparison, Recommendation, StructuralEvent)

GRANULARITY = {"month": 1, "quarter": 2, "year": 3}
LEVEL = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}


def _id(prefix, *parts):
    return prefix + hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:16]


def _date(value):
    return value if isinstance(value, date) else date.fromisoformat(value)


def _mode(values, events=()):
    flags = {value.is_synthetic for value in [*values, *events]}
    if len(flags) > 1:
        raise ValueError("Синтетические и реальные данные нельзя смешивать в одном сравнении.")
    return next(iter(flags), False)


def _source(source):
    return {"source_id": source.id, "document": source.document, "locator": source.locator, "quote": source.text}


def _suggest_metrics(text):
    text = text.casefold().replace("ё", "е")
    result = []
    for roots, metric in ((('отчет', 'отчёт'), 'report_delay_days'), (('качеств',), 'qa_score'),
                         (('покрыти', 'карта гарантий'), 'assurance_coverage'),
                         (('проверк', 'аудит'), 'audits_done'), (('мероприяти',), 'action_overdue')):
        if any(root in text for root in roots):
            result.append(metric)
    return result


def events_from_core(result, effective_date, is_synthetic=False):
    """Build candidates; the user must confirm the date and any mechanism map."""
    effective_date = _date(effective_date).isoformat()
    events = []
    def add(kind, units, description, sources, finding_id=""):
        evidence = list({source.id: _source(source) for source in sources}.values())
        if not evidence:
            return
        units = sorted(set(unit for unit in units if unit and unit != "Владелец не установлен"))
        metrics = _suggest_metrics(description)
        events.append(StructuralEvent(
            _id("ev_", kind, effective_date, units, description), kind, units, effective_date,
            evidence, finding_id, description, metrics,
            "Предполагаемая связь формулировки функции с показателем; требуется подтверждение сотрудником." if metrics else "",
            False, is_synthetic, "user_confirmed_date; structural_candidate"))
    for change in result.unit_changes:
        if change.status in {"created", "removed", "transformed"}:
            kind = {"created": "unit_created", "removed": "unit_abolished", "transformed": "unit_transformed"}[change.status]
            add(kind, [change.before, change.after], f"Изменение перечня: {change.before or '—'} → {change.after or '—'}", change.sources)
    for row in result.matrix_rows:
        if row.norm_type == "prohibition" or row.status not in {"lost", "moved", "changed"}:
            continue
        old_text = " ".join(f.text for f in row.before).casefold()
        new_text = " ".join(f.text for f in row.after).casefold()
        weakened = row.status == "changed" and any(word in old_text and word not in new_text for word in ("ежекварталь", "ежемесяч", "обязан", "должен"))
        kind = "func_lost" if row.status == "lost" else "func_transferred" if row.status == "moved" else "func_weakened" if weakened else "func_changed"
        sources = [s for f in row.before + row.after for s in (f.source, *f.context_sources)]
        add(kind, [f.unit for f in row.before + row.after], "Кандидат изменения закрепления: " + row.label, sources, row.id)
    for finding in result.findings:
        if finding.kind in {"duplicate", "conflict"}:
            row = next((row for row in result.matrix_rows if row.id == finding.matrix_row_id), None)
            units = [f.unit for f in row.before + row.after] if row else []
            add("func_overlap" if finding.kind == "duplicate" else "role_conflict", units,
                "Кандидат для проверки: " + finding.title, finding.sources, finding.matrix_row_id)
    return list({event.id: event for event in events}.values())


@dataclass
class _Point:
    start: date
    end: date
    value: float
    ids: list[str]
    granularity: str


def _bounds(day, granularity):
    month = 1 if granularity == "year" else (day.month - 1) // 3 * 3 + 1 if granularity == "quarter" else day.month
    end_month = 12 if granularity == "year" else month + 2 if granularity == "quarter" else month
    return date(day.year, month, 1), date(day.year, end_month, calendar.monthrange(day.year, end_month)[1])


def _deduplicate(values):
    grouped = {}
    ids_seen, units_seen = {}, defaultdict(set)
    for value in values:
        if isinstance(value.value, bool) or not isinstance(value.value, (int, float)) or not math.isfinite(value.value):
            raise ValueError("Показатель должен быть конечным числом.")
        if not value.id or not value.source.file or not value.source.quote or not (value.source.cell or value.source.locator or value.source.page):
            raise ValueError("У числового показателя нет воспроизводимого источника.")
        start, end = _date(value.period_start), _date(value.period_end)
        if value.granularity not in GRANULARITY or start > end or _bounds(start, value.granularity) != (start, end):
            raise ValueError("Период показателя не соответствует полному месяцу, кварталу или году.")
        identity = (value.metric, value.unit_scope, start, end, value.value, value.unit)
        if value.id in ids_seen and ids_seen[value.id] != identity:
            raise ValueError("Один идентификатор метрики ссылается на разные значения.")
        ids_seen[value.id] = identity
        if value.unit:
            units_seen[value.metric, value.unit_scope].add(value.unit.strip().casefold())
            if len(units_seen[value.metric, value.unit_scope]) > 1:
                raise ValueError("Единицы измерения метрики различаются; сначала подтвердите приведение единиц.")
        key = value.metric, value.unit_scope, start, end
        if key in grouped:
            point = grouped[key]
            if not math.isclose(point.value, value.value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Конфликтующие значения одной метрики, подразделения и периода требуют подтверждения.")
            if value.id not in point.ids:
                point.ids.append(value.id)
        else:
            grouped[key] = _Point(start, end, float(value.value), [value.id], value.granularity)
    return grouped


def _canonical(points, target, aggregation):
    buckets = defaultdict(list)
    for point in points:
        buckets[_bounds(point.start, target)].append(point)
    result, excluded, warnings = [], [], []
    for (start, end), items in sorted(buckets.items()):
        native = [point for point in items if point.start == start and point.end == end]
        if native:
            nested = sorted((point for point in items if point not in native), key=lambda point: (point.start, point.end))
            if nested and aggregation in {'sum', 'last'}:
                contiguous = nested[0].start == start and nested[-1].end == end and all(right.start == left.end + timedelta(days=1) for left, right in zip(nested, nested[1:]))
                if contiguous:
                    reconciled = sum(point.value for point in nested) if aggregation == 'sum' else nested[-1].value
                    if not math.isclose(native[0].value, reconciled, rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError("Готовый итог периода противоречит полному набору вложенных периодов.")
            result.append(native[0])
            if len(items) > 1:
                warnings.append(f"{start}/{end}: использовано готовое значение общего периода; вложенные периоды не прибавлялись.")
            continue
        items.sort(key=lambda point: (point.start, point.end))
        cursor = start
        complete = True
        for point in items:
            if point.start < cursor:
                raise ValueError("Перекрывающиеся периоды без однозначного агрегата требуют подтверждения.")
            if point.start != cursor:
                complete = False
            cursor = point.end + timedelta(days=1)
        if not complete or cursor != end + timedelta(days=1):
            excluded.append(f"{start}/{end}: неполный набор для {target}")
            continue
        numbers = [point.value for point in items]
        value = sum(numbers) if aggregation == "sum" else numbers[-1] if aggregation == "last" else statistics.mean(numbers)
        result.append(_Point(start, end, value, [key for point in items for key in point.ids], target))
    return result, excluded, warnings


def _mean(points):
    return statistics.mean(point.value for point in points)


def _midpoint(point):
    return (point.start.toordinal() + point.end.toordinal()) / 2


def _trend(points):
    x = [_midpoint(point) for point in points]
    y = [point.value for point in points]
    mx, my = statistics.mean(x), statistics.mean(y)
    spread = sum((value - mx) ** 2 for value in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / spread if spread else 0.0
    return lambda point: my + slope * (_midpoint(point) - mx), slope


def _period_summary(points):
    return {"period": f"{points[0].start}/{points[-1].end}", "start": points[0].start.isoformat(),
            "end": points[-1].end.isoformat(), "value": _mean(points), "n_points": len(points),
            "metric_ids": [key for point in points for key in point.ids],
            "points": [{"period_start": point.start.isoformat(), "period_end": point.end.isoformat(),
                        "value": point.value, "metric_ids": point.ids[:]} for point in points]}


def _control_config(controls, event, scope):
    if not isinstance(controls, dict):
        return None
    config = controls.get(event.id, controls.get(scope))
    if not isinstance(config, dict) or config.get("verified") is not True or not str(config.get("reason", "")).strip():
        return None
    target = config.get("scope")
    return target if isinstance(target, str) and target and target != scope and target not in event.units else None


def compare_metrics(values, events, *, control_scopes=None):
    synthetic = _mode(values, events)
    indexed = _deduplicate(values)
    group = defaultdict(list)
    for (metric, scope, _, _), point in indexed.items():
        group[metric, scope].append(point)
    comparisons = []
    for event in events:
        effective = _date(event.effective_date)
        for (metric, scope), raw in sorted(group.items()):
            if scope not in event.units or (event.metric_codes and metric not in event.metric_codes):
                continue
            definition = METRIC_CATALOG.get(metric)
            if definition is None:
                if not all(value.confirmed_by_human for value in values if value.metric == metric):
                    continue
                aggregation = "mean"
            else:
                aggregation = definition.aggregation
            # A verified control has to use the same time buckets too.
            control = _control_config(control_scopes, event, scope)
            control_raw = group.get((metric, control), []) if control else []
            target = max((point.granularity for point in raw), key=GRANULARITY.get)
            control_target = max((point.granularity for point in raw + control_raw), key=GRANULARITY.get)
            if control and control_target != target:
                candidate, _, _ = _canonical(raw, control_target, aggregation)
                candidate_control, _, _ = _canonical(control_raw, control_target, aggregation)
                candidate_before = [point for point in candidate if point.end < effective]
                candidate_after = [point for point in candidate if point.start > effective]
                control_periods = {(point.start, point.end) for point in candidate_control}
                if (candidate_before and candidate_after and all((point.start, point.end) in control_periods for point in candidate_before + candidate_after)
                        and not any(other.id != event.id and control in other.units and candidate_before[0].start <= _date(other.effective_date) <= candidate_after[-1].end for other in events)):
                    target = control_target
            points, excluded, warnings = _canonical(raw, target, aggregation)
            before = [point for point in points if point.end < effective]
            after = [point for point in points if point.start > effective]
            excluded.extend(f"{point.start}/{point.end}: пересекает дату изменения" for point in points if point not in before + after)
            if not before or not after:
                continue
            left, right = _period_summary(before), _period_summary(after)
            delta = right['value'] - left['value']
            percentage = delta / abs(left['value']) * 100 if left['value'] else None
            method, adjusted, control_scope = "simple", None, None
            warnings.append("Статистическая значимость не оценивалась; p-value не рассчитывался. Сравнение не устанавливает причину.")
            if target != min((point.granularity for point in raw), key=GRANULARITY.get):
                warnings.append(f"Приведение к {target}: {aggregation}; сравниваются средние значения сопоставимых периодов, а не суммы окон разной длины.")
                if definition and definition.unit == "%":
                    warnings.append("Проценты усреднены арифметически, не суммировались; числители и знаменатели для взвешивания не предоставлены.")
            control_map = {}
            if control:
                touched = any(other.id != event.id and control in other.units and before[0].start <= _date(other.effective_date) <= after[-1].end for other in events)
                cpoints, _, cwarn = _canonical(control_raw, target, aggregation)
                control_map = {(point.start, point.end): point for point in cpoints}
                exact = all((point.start, point.end) in control_map for point in before + after)
                if not touched and exact and cpoints:
                    old_control = [control_map[point.start, point.end] for point in before]
                    new_control = [control_map[point.start, point.end] for point in after]
                    method, control_scope = "diff_in_diff", control
                    adjusted = delta - (_mean(new_control) - _mean(old_control))
                    left["control_value"], right["control_value"] = _mean(old_control), _mean(new_control)
                    for summary, actual in ((left, before), (right, after)):
                        for record, point in zip(summary['points'], actual):
                            control_point = control_map[point.start, point.end]
                            record["control_value"] = control_point.value
                            record["control_metric_ids"] = control_point.ids[:]
                    if len(before) < 3:
                        warnings.append("Предпосылка параллельных трендов до изменения не проверена: меньше трёх точек.")
                    else:
                        _, slope = _trend(before)
                        _, control_slope = _trend(old_control)
                        if abs(slope - control_slope) * (before[-1].end - before[0].start).days > max(abs(left['value']) * .1, 1):
                            warnings.append("Предпосылка параллельных трендов сомнительна: до изменения динамика групп различается.")
                    warnings.extend(cwarn)
                else:
                    warnings.append("Контроль не использован: затронут другими событиями или нет точно совпадающих периодов.")
            if method == "simple" and len(before) >= 3:
                prediction, _ = _trend(before)
                predicted = [prediction(point) for point in after]
                if any(value < 0 or (definition and definition.unit == '%' and value > 100) for value in predicted):
                    warnings.append("Линейное продолжение вышло за допустимый диапазон; использовано простое сравнение, сезонность не подтверждена.")
                else:
                    adjusted = statistics.mean(point.value - prediction(point) for point in after)
                    method = "trend_adjusted"
                    for record, point in zip(right['points'], after):
                        record['predicted'] = prediction(point)
                    warnings.append("Поправка на линейный тренд использует реальные даты; сезонность и устойчивость продолжения тренда не подтверждены.")
            if method == "simple":
                warnings.append("Простое сравнение не отделяет изменение структуры от прежнего тренда.")
            normalized, normalization_sources = None, []
            if definition and definition.normalize_by:
                fte = [indexed.get((definition.normalize_by, scope, point.start, point.end)) for point in before + after]
                if all(point is not None and point.value > 0 for point in fte):
                    normalized_values = [point.value / denominator.value for point, denominator in zip(before + after, fte)]
                    left['normalized_value'] = statistics.mean(normalized_values[:len(before)])
                    right['normalized_value'] = statistics.mean(normalized_values[len(before):])
                    normalized = right['normalized_value'] - left['normalized_value']
                    normalization_sources = [key for point in fte for key in point.ids]
                else:
                    warnings.append("Нормализация на FTE недоступна: нужны положительные значения того же подразделения и точно того же периода.")
            ids = [key for point in before + after for key in point.ids]
            if method == "diff_in_diff":
                ids.extend(key for point in before + after for key in control_map[point.start, point.end].ids)
            right['effective_date'] = effective.isoformat()
            right['expected_start'] = (_bounds(effective, target)[1] + timedelta(days=1)).isoformat()
            left['granularity'] = right['granularity'] = target
            left['aggregation'] = right['aggregation'] = aggregation
            right['descriptive_change'] = abs(delta) > max(abs(left['value']) * .05, 1e-9)
            comparisons.append(PeriodComparison(_id("cmp_", event.id, metric, scope, left['period'], right['period']),
                metric, scope, event.id, left, right, delta, percentage, normalized, "insufficient_data", method,
                list(dict.fromkeys(ids)), control_scope, adjusted, aggregation, excluded, list(dict.fromkeys(warnings)), synthetic, normalization_sources))
    return comparisons


def calculate_confidence(event, comparison, opponent_outcome, alternative_events=None):
    reasons = []
    if min(comparison.before.get('n_points', 0), comparison.after.get('n_points', 0)) < 2:
        return "insufficient", ["Менее двух сопоставимых точек в одном из периодов."]
    if opponent_outcome == "refuted":
        return "insufficient", ["Оппонент опроверг предложенную связь."]
    if not event.mechanism or comparison.metric not in event.metric_codes:
        return "insufficient", ["Связь функции с метрикой не задана."]
    if not event.mechanism_confirmed:
        return "low", ["Механизм предложен, но не подтверждён сотрудником; прямой механизм не установлен."]
    if opponent_outcome == "upheld" and any('Предпосылка' in warning or 'сезонность' in warning for warning in comparison.warnings):
        opponent_outcome = "weakened"
        reasons.append("Предпосылки выбранного метода не подтверждены; высокий уровень не присваивается.")
    if opponent_outcome == "weakened" and comparison.method == "simple":
        return "low", reasons + ["Простое сравнение и замечания оппонента не позволяют отделить альтернативы."]
    reasons.append("Связь функции и метрики подтверждена сотрудником.")
    reasons.append("Метод сравнения: " + comparison.method + ".")
    if comparison.method in {"diff_in_diff", "trend_adjusted"} and opponent_outcome == "upheld" and not alternative_events:
        return "high", reasons + ["Существенных альтернатив в предоставленных данных не обнаружено; это не доказательство причины."]
    if opponent_outcome in {"upheld", "weakened"}:
        return "medium", reasons + ["Альтернативы или ограничения метода сохраняются; связь требует экспертной проверки."]
    return "low", reasons + ["Независимый разбор альтернатив не завершён."]


def attribute_comparisons(values, events, comparisons):
    _mode(values, [*events, *comparisons])
    event_by_id = {event.id: event for event in events}
    indexed = _deduplicate(values)
    results = []
    for comparison in comparisons:
        event = event_by_id.get(comparison.event_id)
        if event is None or not comparison.after.get('descriptive_change', abs(comparison.delta_abs) > 1e-9):
            continue
        effective = _date(event.effective_date)
        neighbors = [other for other in events if other.id != event.id and
                     (_date(comparison.before['start']) <= _date(other.effective_date) <= _date(comparison.after['end']) or
                      abs((_date(other.effective_date) - effective).days) <= 90)]
        alternatives = [f"Событие {other.id}: {other.description or other.type} ({other.effective_date}; {', '.join(other.units) or 'область не определена'})." for other in neighbors]
        relevant = [other for other in neighbors if not other.units or set(other.units) & set(event.units) or "BVA" in other.units or "БВА" in other.units]
        staff_ids = []
        for metric in ('fte_actual', 'headcount', 'audits_planned', 'budget_fact'):
            raw = [point for (code, scope, _, _), point in indexed.items() if code == metric and scope == comparison.unit_scope
                   and point.start >= _date(comparison.before['start']) and point.end <= _date(comparison.after['end'])]
            if raw:
                raw, _, _ = _canonical(raw, comparison.before.get('granularity', raw[0].granularity), METRIC_CATALOG[metric].aggregation)
            before = [point for point in raw if point.end < effective]
            after = [point for point in raw if point.start > effective]
            if before and after and not math.isclose(_mean(before), _mean(after), rel_tol=.05, abs_tol=1e-9):
                label = 'найм/изменение фактического штата' if metric == 'fte_actual' else 'изменение штатной численности' if metric == 'headcount' else 'изменение плана или бюджета'
                alternatives.append(f"{label}: {metric} до {_mean(before):.4g}, после {_mean(after):.4g}; требуется проверка сопоставимости.")
                staff_ids.extend(key for point in before + after for key in point.ids)
        limitations = [warning for warning in comparison.warnings if 'Предпосылка' in warning or 'сезонность' in warning]
        alternatives.extend(limitations)
        if comparison.method == 'trend_adjusted' and comparison.adjusted_delta is not None and abs(comparison.adjusted_delta) < abs(comparison.delta_abs) * .2:
            alternatives.append("Большая часть наблюдаемого изменения согласуется с продолжением прежнего тренда.")
            outcome = 'refuted'
        else:
            outcome = 'weakened' if relevant or staff_ids or limitations else 'upheld'
        confidence, reasons = calculate_confidence(event, comparison, outcome, neighbors or staff_ids or limitations)
        direction = 'вырос' if comparison.delta_abs > 0 else 'снизился' if comparison.delta_abs < 0 else 'не изменился'
        definition = METRIC_CATALOG.get(comparison.metric)
        label = definition.name if definition else comparison.metric
        claim = f"После изменения «{event.description or event.type}» показатель «{label}» ({comparison.unit_scope}) {direction}: {comparison.before['value']:.4g} → {comparison.after['value']:.4g}. Это наблюдаемая связь, не установленная причина."
        if definition and definition.better == 'context':
            claim += " Рост или снижение количества находок нельзя автоматически считать улучшением либо ухудшением."
        results.append(Attribution(_id('att_', event.id, comparison.id), event.id, comparison.id, claim,
            event.mechanism or 'Механизм не задан; необходима проверка сотрудником.', confidence, reasons,
            alternatives or ["Других событий и заметных ресурсных изменений в предоставленных данных не найдено; непредоставленные причины не исключены."],
            {'clauses': event.evidence, 'metrics': list(dict.fromkeys(comparison.metric_ids + comparison.normalization_sources + staff_ids)), 'alternative_events': [other.id for other in neighbors]},
            outcome, 'Детерминированный оппонент проверил соседние события, кадровые показатели, план, бюджет и ограничения тренда; внешние причины не исследовались.',
            comparison.is_synthetic, 'rule'))
    return results


def check_hypotheses(hypotheses, comparisons, *, as_of=None):
    today = _date(as_of) if as_of else date.today()
    checked = []
    for original in hypotheses:
        hypothesis = replace(original, checked_with=[])
        horizon = _date(hypothesis.check_after)
        if today < horizon:
            hypothesis.status, hypothesis.explanation = 'pending', 'Горизонт проверки ещё не наступил.'
            checked.append(hypothesis)
            continue
        candidates = [comparison for comparison in comparisons if comparison.metric == hypothesis.metric and comparison.unit_scope == hypothesis.unit_scope
                      and (not hypothesis.event_id or comparison.event_id == hypothesis.event_id) and comparison.is_synthetic == hypothesis.is_synthetic]
        directions = []
        for comparison in candidates:
            points = [point for point in comparison.after.get('points', []) if _date(point['period_end']) <= horizon]
            points.sort(key=lambda point: point['period_start'])
            if len(points) < 2 or comparison.before.get('n_points', 0) < 2 or _date(points[-1]['period_end']) < horizon:
                continue
            expected_start = comparison.after.get('expected_start', comparison.after.get('start'))
            if expected_start and _date(points[0]['period_start']) != _date(expected_start):
                continue
            if any(_date(right['period_start']) != _date(left['period_end']) + timedelta(days=1) for left, right in zip(points, points[1:])):
                continue
            delta = statistics.mean(point['value'] for point in points) - comparison.before['value']
            directions.append('up' if delta > 1e-9 else 'down' if delta < -1e-9 else 'stable')
            hypothesis.checked_with.append(comparison.id)
        if not directions or len(set(directions)) != 1:
            hypothesis.status, hypothesis.explanation = 'inconclusive', 'Нет достаточного непрерывного покрытия до горизонта проверки либо сопоставления противоречат друг другу.'
        else:
            hypothesis.status = 'confirmed' if directions[0] == hypothesis.expected_direction else 'refuted'
            hypothesis.explanation = 'Направление наблюдаемого изменения проверено по данным до указанного горизонта; это не подтверждает причинный механизм.'
        checked.append(hypothesis)
    return checked


def hypothesis_accuracy(hypotheses):
    counts = Counter(hypothesis.status for hypothesis in hypotheses)
    checked = counts['confirmed'] + counts['refuted']
    return {key: counts[key] for key in ('confirmed', 'refuted', 'inconclusive', 'pending')} | {'checked': checked, 'accuracy': counts['confirmed'] / checked if checked else None}


def recommend_actions(events, attributions, comparisons, knowledge, *, core_result=None):
    event_by_id = {event.id: event for event in events}
    comparison_by_id = {comparison.id: comparison for comparison in comparisons}
    recommendations = []
    for attribution in attributions:
        event, comparison = event_by_id.get(attribution.event_id), comparison_by_id.get(attribution.comparison_id)
        if event is None or comparison is None or not event.units or attribution.confidence == 'insufficient':
            continue
        definition = METRIC_CATALOG.get(comparison.metric)
        if definition is None or definition.better not in {'up', 'down'}:
            continue
        adverse = comparison.delta_abs < 0 if definition.better == 'up' else comparison.delta_abs > 0
        if not adverse:
            continue
        records = [record for record in knowledge if record.is_synthetic == attribution.is_synthetic and record.observed_effect.get('metric') == comparison.metric
                   and LEVEL.get(record.confidence, 0) >= 2
                   and record.change_pattern == f"{event.type}: {' '.join(event.mechanism.casefold().split())}"]
        action_type = 'restore_function' if event.type in {'func_lost', 'func_weakened'} else 'staffing' if event.type in {'staff_cut', 'staff_hired', 'staff_changed'} else 'reassign_function'
        action = ('Проверить необходимость восстановления или уточнения закрепления функции' if action_type == 'restore_function' else
                  'Проверить нагрузку и достаточность фактических ресурсов' if action_type == 'staffing' else 'Проверить разграничение существующих обязанностей и областей ответственности')
        action += ' в подразделении ' + ', '.join(event.units) + '; решение принимает сотрудник.'
        expected = {'metric': comparison.metric, 'direction': definition.better, 'description': 'Желаемое направление изменения; размер будущего эффекта не прогнозируется.'}
        numeric_cases = []
        for record in records:
            cases = record.context.get('cases', [])
            diversity = {(case.get('event_date'), case.get('unit')) for case in cases}
            numbers = [case.get('delta_pct') for case in cases]
            if (record.n_cases >= 3 and record.diversity >= 3 and len(cases) >= 3 and len(diversity) >= 3
                    and all(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) for number in numbers)):
                numeric_cases.extend(numbers)
        if numeric_cases:
            expected.update(historical_observed_pct_range=[min(numeric_cases), max(numeric_cases)],
                            historical_cases=len(numeric_cases),
                            historical_note='Диапазон наблюдений для исходного изменения, не обещание эффекта предлагаемого действия.')
        recommendations.append(Recommendation(_id('rec_', attribution.id, comparison.metric), action, action_type,
            {'attributions': [attribution.id], 'kb': [record.id for record in records], 'findings': [event.finding_id] if event.finding_id else []},
            expected,
            attribution.confidence, ['Связь не является доказанной причиной; действие может не изменить показатель.', 'Проверить действующие полномочия и независимость контроля до изменения документов.'],
            event.units[:], None, True, 'discuss', attribution.is_synthetic))
    used_events = {attribution.event_id for attribution in attributions if any(attribution.id in recommendation.based_on.get('attributions', []) for recommendation in recommendations)}
    for event in events:
        if event.type not in {'func_overlap', 'role_conflict'} or not event.evidence or event.id in used_events:
            continue
        recommendations.append(Recommendation(_id('rec_core_', event.id),
            'Проверить разграничение существующих обязанностей и независимость проверки; при подтверждении уточнить закрепление ответственности.',
            'reassign_function', {'events': [event.id], 'findings': [event.finding_id] if event.finding_id else [], 'clauses': event.evidence},
            {'metric': None, 'direction': 'unknown', 'description': 'Изменение метрики не оценено; рекомендация основана только на документах ядра.'},
            'low', ['Индикатор из документов не доказывает фактический конфликт или дубль.', 'Новое назначение должно быть обосновано существующими полномочиями и согласовано сотрудником.'],
            event.units[:], None, True, 'discuss', event.is_synthetic))
    return recommendations
