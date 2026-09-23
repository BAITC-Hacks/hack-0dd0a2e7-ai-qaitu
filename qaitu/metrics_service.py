"""Explicit, optional second-layer analysis; never runs during core comparison."""
from __future__ import annotations

from datetime import date
import json
import math
import os
from pathlib import Path
import uuid

from .metrics_models import MetricsRun, StructuralEvent, Hypothesis
from .metrics_store import DEFAULT_ROOT, persist_run, update_knowledge, restore_decisions, save_decision, load_knowledge, cache_mechanisms, apply_cached_mechanisms


def metrics_enabled():
    return os.environ.get("METRICS_MODULE", "on").casefold() not in {"off", "0", "false", "no"}


def events_from_core(result, effective_date, is_synthetic=False):
    from .metrics_analysis import events_from_core as build
    return apply_cached_mechanisms(build(result, effective_date, is_synthetic=is_synthetic))


def ingest_reports(files, *, unit_scope=None, is_synthetic=False, use_ai=False):
    from .metrics_ingest import ingest_reports as ingest
    extractor = None
    if use_ai:
        from .metrics_llm import extract_metrics
        extractor = extract_metrics
    return ingest(files, unit_scope=unit_scope, is_synthetic=is_synthetic, text_extractor=extractor)


def _validate(values, events, hypotheses):
    if not values:
        raise ValueError("Нет подтверждённых числовых значений для сравнения")
    flags = {v.is_synthetic for v in values} | {e.is_synthetic for e in events} | {h.is_synthetic for h in hypotheses}
    if len(flags) != 1 or not all(type(flag) is bool for flag in flags):
        raise ValueError("Нельзя смешивать реальные и тестовые данные в одном запуске")
    ids = set()
    for value in values:
        if value.id in ids:
            raise ValueError("Повторный идентификатор значения: разрешите дубликаты перед расчётом")
        ids.add(value.id)
        source = value.source
        if not source.file or not source.quote or not (source.cell or source.page or source.locator):
            raise ValueError("У каждого значения должен быть источник: файл, цитата и точное место")
        if isinstance(value.value, bool) or not isinstance(value.value, (int, float)) or not math.isfinite(value.value):
            raise ValueError("Числовые значения должны быть конечными")
        if not value.unit_scope or not value.metric or date.fromisoformat(value.period_start) > date.fromisoformat(value.period_end):
            raise ValueError("Проверьте подразделение, метрику и период")
    if len({event.id for event in events}) != len(events):
        raise ValueError("Повторный идентификатор структурного события")
    for event in events:
        date.fromisoformat(event.effective_date)
        if event.mechanism_confirmed and (not event.evidence or not event.mechanism.strip()):
            raise ValueError("Подтверждённому механизму нужны цитаты и описание")
    return next(iter(flags))


def run_metrics(values, events, *, hypotheses=None, control_scopes=None, use_ai=False, as_of=None, storage_root=None, custom_definitions=None):
    if not metrics_enabled():
        raise ValueError("Модуль метрик отключён (METRICS_MODULE=off)")
    from .metrics_analysis import compare_metrics, attribute_comparisons, check_hypotheses, recommend_actions
    hypotheses = list(hypotheses or [])
    synthetic = _validate(values, events, hypotheses)
    run = MetricsRun(run_id=f"metrics_{uuid.uuid4().hex}", is_synthetic=synthetic,
                     values=list(values), events=list(events), custom_definitions=list(custom_definitions or []),
                     settings={"as_of": str(as_of or date.today()), "use_ai": use_ai, "control_scopes": control_scopes or {},
                               "human_review_required": True, "statistical_significance_estimated": False})
    cache_mechanisms(run.events, storage_root)
    run.comparisons = compare_metrics(run.values, run.events, control_scopes=control_scopes)
    run.attributions = attribute_comparisons(run.values, run.events, run.comparisons)
    run.warnings = list(dict.fromkeys(warning for comparison in run.comparisons for warning in comparison.warnings))
    if use_ai:
        from .metrics_llm import enhance_attributions
        run.attributions, warnings = enhance_attributions(run.values, run.events, run.comparisons, run.attributions)
        run.warnings.extend(warnings)
    run.hypotheses = check_hypotheses(hypotheses, run.comparisons, as_of=as_of)
    run.knowledge = update_knowledge(run, storage_root)
    run.recommendations = recommend_actions(run.events, run.attributions, run.comparisons, run.knowledge)
    if use_ai:
        from .metrics_llm import refine_recommendations
        run.recommendations, warnings = refine_recommendations(run.recommendations, run.attributions, run.knowledge)
        run.warnings.extend(warnings)
    if not run.events:
        run.warnings.append("Структурные события не заданы: доступны исходные ряды, связь с реорганизацией не оценивалась.")
    if not run.hypotheses:
        run.warnings.append("В текущем ядре нет таймлайна или симулятора; прогнозы не созданы автоматически.")
    if synthetic:
        run.warnings.insert(0, "ТЕСТОВЫЕ ДАННЫЕ: эффекты заложены в генераторе и не описывают реальные результаты компании.")
    restore_decisions(run, storage_root)
    persist_run(run, storage_root)
    return run


def demo_inputs():
    from scripts.gen_synthetic_reports import generate_reports
    manifest = generate_reports(DEFAULT_ROOT / "data" / "synthetic_reports")
    paths = manifest.get("files", manifest.get("paths", []))
    result = ingest_reports([(Path(p).name, Path(p).read_bytes()) for p in paths], is_synthetic=True)
    events = [StructuralEvent(**event) if isinstance(event, dict) else event for event in manifest["events"]]
    hypotheses = [Hypothesis(**h) if isinstance(h, dict) else h for h in manifest.get("hypotheses", [])]
    return result, events, hypotheses, manifest.get("controls", {})


def render_markdown(run: MetricsRun):
    from .metrics_analysis import hypothesis_accuracy
    label = "ТЕСТОВЫЕ ДАННЫЕ" if run.is_synthetic else "ДАННЫЕ ОТЧЁТОВ"
    lines = []
    def add(text=""):
        # Explicit label on every exported row, including headings and tables.
        lines.append(f"{label} · {str(text).replace(chr(10), ' / ')}")
    add("QAITU — оценка эффективности и оптимизация")
    add(f"Запуск: {run.run_id}")
    add("Выводы рекомендательные. Наблюдаемая связь не доказывает причину. Статистическая значимость не оценивалась.")
    for warning in run.warnings:
        add(warning)
    for definition in run.custom_definitions:
        add(f"Пользовательский показатель: {definition.code} — {definition.name}; {definition.unit}; подтверждено: {definition.confirmed_by_human}")
    add("ИСТОЧНИКИ ЗНАЧЕНИЙ")
    for value in run.values:
        source = value.source
        locator = " / ".join(str(x) for x in (source.sheet, source.cell, f"стр. {source.page}" if source.page else None, source.locator) if x)
        add(f"{value.id} | {value.metric} | {value.unit_scope} | {value.period_start}–{value.period_end} | {value.value:g} {value.unit} | {source.file} / {locator} | «{source.quote}»")
    add("СОБЫТИЯ И СРАВНЕНИЯ")
    for event in run.events:
        add(f"{event.id}: {event.description} | {event.effective_date} | основание даты: {event.date_basis}")
        for evidence in event.evidence:
            add(json.dumps(evidence, ensure_ascii=False))
    for comparison in run.comparisons:
        add(f"{comparison.id} | {comparison.metric}, {comparison.unit_scope} | {comparison.method} | агрегация: {comparison.aggregation}")
        add(f"До: {comparison.before.get('value')} ({comparison.before.get('n_points')} точек); после: {comparison.after.get('value')} ({comparison.after.get('n_points')} точек); разница: {comparison.delta_abs:g}; относительная разница: {comparison.delta_pct}; скорректированная разница: {comparison.adjusted_delta}")
        add(f"Нормализация на FTE: {comparison.normalized_delta}; источники: {', '.join(comparison.metric_ids + comparison.normalization_sources)}")
        add(f"Исключённые переходные периоды: {comparison.excluded_periods}")
    add("ИЗМЕНЕНИЕ → ЭФФЕКТ")
    for attribution in run.attributions:
        add(f"{attribution.id}: {attribution.claim} | уверенность: {attribution.confidence} | способ: {attribution.method}")
        add(f"Механизм: {attribution.mechanism}; основания: {'; '.join(attribution.confidence_reasons)}")
        add(f"Альтернативные объяснения: {'; '.join(attribution.alternative_explanations)}")
        add(f"Оппонент: {attribution.opponent_outcome}; {attribution.opponent_reasoning}")
        add(f"Источники: {json.dumps(attribution.evidence, ensure_ascii=False)}")
    add("ПРОГНОЗ VS ФАКТ")
    accuracy = hypothesis_accuracy([h for h in run.hypotheses if h.source_type != "manual"])
    add(f"Точность прогнозов агента (ручные исключены): {accuracy.get('accuracy')}; проверено с определённым исходом: {accuracy.get('checked')}; остальные прогнозы исключены из знаменателя")
    for h in run.hypotheses:
        add(f"{h.id} | {h.source_type} | {h.prediction} | срок: {h.check_after} | {h.status} | {h.explanation} | {h.checked_with}")
    add("РЕКОМЕНДАЦИИ — РЕШЕНИЕ ПРИНИМАЕТ ЧЕЛОВЕК")
    for recommendation in run.recommendations:
        add(f"{recommendation.id}: {recommendation.action}; уверенность: {recommendation.confidence}; решение: {recommendation.decision}")
        add(f"Ожидаемое направление: {json.dumps(recommendation.expected_effect, ensure_ascii=False)}; риски: {'; '.join(recommendation.risks)}; основания: {recommendation.based_on}")
    add("БАЗА ЗНАНИЙ")
    for record in run.knowledge:
        add(f"{record.id}: {record.change_pattern}; случаев: {record.n_cases}; {record.context.get('interpretation', '')}; эффект: {record.observed_effect}")
    return "\n\n".join(lines)
