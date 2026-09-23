"""Optional metrics workspace. Imported only when the workspace is selected."""
from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from datetime import date
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

from .metrics_catalog import METRIC_CATALOG
from .config import load_openai_settings
from .metrics_ingest import confirm_ambiguity
from .metrics_models import Hypothesis, IngestionResult
from . import metrics_service as service

TEST_LABEL = "ТЕСТОВЫЕ ДАННЫЕ"
REAL_LABEL = "РЕАЛЬНЫЕ ОТЧЁТЫ"
CONFIDENCE = {"high": "Высокая", "medium": "Средняя", "low": "Низкая", "insufficient": "Недостаточно данных"}
METHODS = {"simple": "Простое сравнение", "trend_adjusted": "Поправка на тренд", "diff_in_diff": "Разность разностей"}
HYPOTHESES = {"pending": "⏳ Ожидает проверки", "confirmed": "✅ Подтверждён", "refuted": "❌ Не подтвердился", "inconclusive": "Недостаточно данных"}
REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"


def data_label(synthetic: bool) -> str:
    return TEST_LABEL if synthetic else REAL_LABEL


def _date(value, fallback=None):
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return fallback or date.today()


def _metric_name(code):
    definition = METRIC_CATALOG.get(code)
    return definition.name if definition else str(code)


def _source_location(source):
    return " · ".join(str(value) for value in (source.file, source.sheet, source.cell, f"стр. {source.page}" if source.page else None, source.locator) if value)


def value_rows(values):
    return [{"Данные": data_label(value.is_synthetic), "ID": value.id, "Показатель": _metric_name(value.metric), "Код": value.metric,
             "Значение": value.value, "Единица": value.unit, "Подразделение": value.unit_scope, "Начало": value.period_start,
             "Конец": value.period_end, "Источник": _source_location(value.source), "Подтверждено человеком": value.confirmed_by_human} for value in values]


def _show_value(value):
    st.caption(data_label(value.is_synthetic) + " · " + _metric_name(value.metric))
    st.text(f"{value.value:g} {value.unit} · {value.unit_scope} · {value.period_start} — {value.period_end}")
    st.text(_source_location(value.source))
    st.text(value.source.quote)


def _same_kind(ingestion, synthetic):
    records = list(ingestion.values) + list(ingestion.ambiguous)
    if any(item.is_synthetic != synthetic for item in records):
        raise ValueError("Реальные и тестовые данные нельзя смешивать. Переключите источник данных и загрузите отдельный комплект.")


def _invalidate(prefix):
    st.session_state.pop(prefix + "_run", None)


def _show_error(exc, message):
    # Local service validation uses ValueError; provider errors are sanitized by the service.
    st.error(str(exc) if isinstance(exc, ValueError) else f"{message} ({type(exc).__name__}). Исходные данные сохранены.")


def _render_queue(ingestion, prefix, synthetic):
    if not ingestion.ambiguous:
        st.caption("Очередь сопоставления пуста.")
        return
    st.warning(f"Требуют подтверждения: {len(ingestion.ambiguous)}. До подтверждения эти строки не участвуют в расчётах.")
    by_id = {item.id: item for item in ingestion.ambiguous}
    selected = st.selectbox("Подтвердить сопоставление", list(by_id), format_func=lambda key: data_label(by_id[key].is_synthetic) + " · " + by_id[key].reason, key=prefix + "_ambiguity")
    item = by_id[selected]
    proposed = item.proposed
    st.caption(data_label(item.is_synthetic) + " · " + _source_location(item.source))
    st.text(item.source.quote)
    with st.form(prefix + "_mapping_" + selected):
        codes = list(METRIC_CATALOG) + [definition.code for definition in ingestion.custom_definitions if definition.code not in METRIC_CATALOG]
        proposed_code = proposed.get("metric") or ""
        if proposed_code and proposed_code not in codes:
            codes.append(proposed_code)
        codes.append("Новый показатель")
        chosen = st.selectbox("Показатель", codes, index=codes.index(proposed_code) if proposed_code in codes else len(codes) - 1, format_func=lambda code: _metric_name(code))
        custom_code = st.text_input("Код нового показателя (custom_…)", value=proposed_code if proposed_code.startswith("custom_") else "custom_metric")
        custom_name = st.text_input("Описание нового показателя", value=proposed.get("name", ""))
        number = proposed.get("value")
        number = float(number) if isinstance(number, (float, int)) else None
        value = st.number_input("Число из исходной цитаты", value=number)
        scope = st.text_input("Подразделение / область показателя", value=proposed.get("unit_scope") or "")
        unit = st.text_input("Единица измерения", value=proposed.get("unit") or "")
        first, last = st.columns(2)
        start = first.date_input("Начало периода", value=_date(proposed.get("period_start")))
        end = last.date_input("Конец периода", value=_date(proposed.get("period_end")))
        granularities = ["month", "quarter", "year"]
        granularity = st.selectbox("Периодичность", granularities, index=granularities.index(proposed.get("granularity")) if proposed.get("granularity") in granularities else 0, format_func=lambda key: {"month": "Месяц", "quarter": "Квартал", "year": "Год"}[key])
        st.caption("Подтвердите период и владельца по исходному отчёту. Число должно присутствовать в цитате; приложение проверит это.")
        confirmed = st.checkbox("Я сверил показатель, число, подразделение и период с источником")
        submitted = st.form_submit_button("Сохранить сопоставление")
    if submitted:
        if not confirmed or not scope.strip() or value is None or end < start:
            st.error("Заполните область, число и корректный период; подтвердите сверку с источником.")
        else:
            try:
                confirm_ambiguity(ingestion, selected, metric=custom_code.strip() if chosen == "Новый показатель" else chosen, unit_scope=scope.strip(), period_start=start.isoformat(), period_end=end.isoformat(), granularity=granularity, value=value, unit=unit.strip() or None, custom_name=custom_name.strip() or None)
                _invalidate(prefix)
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(str(exc))


def _render_events(events, controls, ingestion, prefix, synthetic, core_result, core_is_synthetic):
    with st.expander("Связать показатели со структурными изменениями", expanded=not events):
        if not synthetic:
            if core_result is None:
                st.info("Для связи с изменениями сначала сравните положения в разделе «Структура и функции». Графики отчётов доступны и без событий.")
            elif core_is_synthetic:
                st.warning("В ядре открыт синтетический пример. Для реальных отчётов сначала сравните реальные положения; смешивание отключено.")
            else:
                effective = st.date_input("Дата вступления изменений в силу", key=prefix + "_core_date")
                verified = st.checkbox("Дата проверена по положению или приказу", key=prefix + "_date_verified")
                st.caption("Дата утверждения и дата вступления в силу могут различаться. При необходимости уточните даты отдельных событий ниже.")
                if st.button("Получить события из сравнения положений", disabled=not verified, key=prefix + "_import_events"):
                    st.session_state[prefix + "_events"] = service.events_from_core(core_result, effective.isoformat(), is_synthetic=False)
                    _invalidate(prefix)
                    st.rerun()
        if not events:
            return
        if not synthetic and load_openai_settings().api_key:
            st.caption("ИИ может предложить связи функций с метриками по цитатам событий. Отправка в OpenAI происходит по кнопке; предложения затем подтверждает человек.")
            if st.button("Предложить связи с ИИ", key=prefix + "_propose_mechanisms"):
                from .metrics_llm import propose_mechanisms

                try:
                    with st.spinner("Предлагаем механизмы по исходным цитатам…"):
                        updated, warnings = propose_mechanisms(events, [asdict(item) for item in METRIC_CATALOG.values()])
                    st.session_state[prefix + "_events"] = updated
                    st.session_state[prefix + "_mechanism_warnings"] = warnings
                    _invalidate(prefix)
                    st.rerun()
                except Exception as exc:
                    _show_error(exc, "Предложение связей не завершено")
        for warning in st.session_state.get(prefix + "_mechanism_warnings", []):
            st.warning(warning)
        event_by_id = {event.id: event for event in events}
        chosen = st.selectbox("Событие для проверки", list(event_by_id), format_func=lambda key: data_label(event_by_id[key].is_synthetic) + " · " + (event_by_id[key].description or key), key=prefix + "_event")
        event = event_by_id[chosen]
        st.caption(data_label(event.is_synthetic) + " · " + event.type)
        for evidence in event.evidence:
            st.text(" · ".join(str(evidence.get(key, "")) for key in ("document", "file", "clause_id", "source_id", "locator") if evidence.get(key)))
            st.text(str(evidence.get("quote", evidence.get("text", ""))))
        scopes = sorted({value.unit_scope for value in ingestion.values} | set(event.units))
        with st.form(prefix + "_event_form_" + chosen):
            event_date = st.date_input("Дата события", value=_date(event.effective_date))
            date_confirmed = st.checkbox("Подтверждаю дату события", value=event.date_basis in {"user_confirmed", "synthetic"})
            units = st.multiselect("Затронутые подразделения — названия в отчётах", scopes, default=event.units)
            codes = sorted(set(METRIC_CATALOG) | {value.metric for value in ingestion.values} | set(event.metric_codes))
            metrics = st.multiselect("Метрики с предполагаемой связью", codes, default=event.metric_codes, format_func=_metric_name)
            mechanism = st.text_area("Механизм связи, основанный на цитате", value=event.mechanism)
            mechanism_confirmed = st.checkbox("Связь функции с этими метриками проверена человеком", value=event.mechanism_confirmed)
            previous_control = controls.get(chosen, {})
            if isinstance(previous_control, str):
                previous_control = {"scope": previous_control, "verified": False}
            control_options = ["Без контрольной группы"] + scopes
            selected_control = previous_control.get("scope", "Без контрольной группы")
            control = st.selectbox("Контрольная группа", control_options, index=control_options.index(selected_control) if selected_control in control_options else 0)
            control_reason = st.text_input("Почему контрольная группа не затронута изменением", value=previous_control.get("reason", ""))
            control_verified = st.checkbox("Незатронутость контрольной группы проверена", value=previous_control.get("verified", False))
            save = st.form_submit_button("Подтвердить настройки события")
        if save:
            if not date_confirmed or not units:
                st.error("Подтвердите дату и выберите затронутые подразделения.")
            elif mechanism_confirmed and (not mechanism.strip() or not metrics or not event.evidence):
                st.error("Для подтверждённого механизма нужны цитата, объяснение и связанные метрики.")
            elif control != "Без контрольной группы" and (control in units or not control_verified or not control_reason.strip()):
                st.error("Контроль должен отличаться от затронутых подразделений; укажите и подтвердите основание выбора.")
            else:
                event.effective_date, event.date_basis = event_date.isoformat(), "user_confirmed"
                event.units, event.metric_codes = units, metrics
                event.mechanism, event.mechanism_confirmed = mechanism.strip(), mechanism_confirmed
                if control == "Без контрольной группы":
                    controls.pop(chosen, None)
                else:
                    controls[chosen] = {"scope": control, "verified": True, "reason": control_reason.strip()}
                _invalidate(prefix)
                st.success("Настройки сохранены. Пересчитайте результаты по кнопке выше.")


def _render_dashboard(run):
    synthetic = run.is_synthetic
    label = data_label(synthetic)
    if not run.values:
        st.info("Пока нет чисел с подтверждёнными источниками.")
        return
    metric_codes = sorted({value.metric for value in run.values})
    default_metric = "report_delay_days" if "report_delay_days" in metric_codes else (run.comparisons[0].metric if run.comparisons and run.comparisons[0].metric in metric_codes else metric_codes[0])
    metric = st.selectbox("Метрика на графике", metric_codes, index=metric_codes.index(default_metric), format_func=_metric_name, key="metrics_chart_metric_" + label)
    selected = [value for value in run.values if value.metric == metric]
    controls = {comparison.control_scope for comparison in run.comparisons if comparison.control_scope}
    data = pd.DataFrame([{"Период": value.period_end, "Значение": value.value, "Подразделение": value.unit_scope,
                          "Роль": "Контрольная группа" if value.unit_scope in controls else "Наблюдаемое подразделение",
                          "Данные": data_label(value.is_synthetic), "Источник": _source_location(value.source)} for value in selected])
    units = sorted({value.unit for value in selected if value.unit})
    chart = alt.Chart(data).mark_line(point=True).encode(x=alt.X("Период:T", title="Отчётный период"), y=alt.Y("Значение:Q", title="Значение" + (", " + units[0] if len(units) == 1 else "")), color=alt.Color("Подразделение:N", legend=alt.Legend(orient="bottom", columns=2, labelLimit=160)), strokeDash=alt.StrokeDash("Роль:N", legend=alt.Legend(orient="bottom", columns=1, labelLimit=300)), tooltip=["Данные:N", "Период:T", "Подразделение:N", "Значение:Q", "Источник:N", "Роль:N"])
    if run.events:
        event_data = pd.DataFrame([{"Дата": event.effective_date, "Событие": event.description or event.id, "Данные": data_label(event.is_synthetic)} for event in run.events])
        rules = alt.Chart(event_data).mark_rule(color="#b08b38", strokeDash=[5, 4]).encode(x="Дата:T", tooltip=["Данные:N", "Дата:T", "Событие:N"])
        chart = chart + rules
    st.caption(label + " · Контрольные группы выделены типом линии; вертикальные линии — даты событий.")
    st.altair_chart(chart.properties(title=alt.TitleParams(text=label, subtitle=textwrap.wrap(_metric_name(metric), width=48), anchor="start"), height=300), width="stretch")
    if metric in {"findings_total", "findings_material"}:
        st.caption("Рост числа найденных нарушений имеет неоднозначный смысл: это может отражать как более тщательный аудит, так и ухудшение процессов.")
    if run.comparisons:
        st.dataframe(pd.DataFrame([{"Данные": data_label(item.is_synthetic), "Метрика": _metric_name(item.metric), "Подразделение": item.unit_scope, "Метод": METHODS.get(item.method, item.method), "Агрегация": item.aggregation, "До": item.before.get("value"), "Точек до": item.before.get("n_points"), "После": item.after.get("value"), "Точек после": item.after.get("n_points"), "Изменение": item.delta_abs, "Изменение, %": item.delta_pct, "Поправка метода": item.adjusted_delta, "Изменение на FTE": item.normalized_delta, "Исключены периоды": ", ".join(item.excluded_periods), "Предупреждения": "; ".join(item.warnings)} for item in run.comparisons]), hide_index=True, width="stretch")


def _render_attributions(run):
    values = {value.id: value for value in run.values}
    events = {event.id: event for event in run.events}
    comparisons = {item.id: item for item in run.comparisons}
    if not run.attributions:
        st.info("Связи не построены: нужны показатели до/после, структурные события и проверяемый механизм.")
    for item in run.attributions:
        with st.expander(data_label(item.is_synthetic) + " · " + item.claim):
            st.caption("Уверенность: " + CONFIDENCE.get(item.confidence, item.confidence))
            st.write(item.mechanism)
            for reason in item.confidence_reasons:
                st.write("Основание уровня: " + reason)
            comparison = comparisons.get(item.comparison_id)
            if comparison:
                st.caption(data_label(comparison.is_synthetic) + " · " + METHODS.get(comparison.method, comparison.method))
                st.text(f"До: {comparison.before.get('value')} ({comparison.before.get('n_points')} точек) · После: {comparison.after.get('value')} ({comparison.after.get('n_points')} точек) · Разница: {comparison.delta_abs:g}")
                st.caption("Агрегация: " + comparison.aggregation)
                if comparison.excluded_periods:
                    st.caption("Переходные/несопоставимые периоды исключены: " + ", ".join(comparison.excluded_periods))
            st.markdown("**Альтернативные объяснения**")
            for explanation in item.alternative_explanations:
                st.write(explanation)
            st.caption("Вердикт оппонента: " + {"upheld": "связь выдержала проверку", "weakened": "связь ослаблена", "refuted": "связь не подтверждается"}.get(item.opponent_outcome, item.opponent_outcome))
            st.write(item.opponent_reasoning)
            event = events.get(item.event_id)
            if event:
                st.markdown("**Положение / приказ**")
                for source in event.evidence:
                    st.text(" · ".join(str(source.get(key, "")) for key in ("document", "file", "clause_id", "source_id", "locator") if source.get(key)))
                    st.text(str(source.get("quote", source.get("text", ""))))
            st.markdown("**Числа и первоисточники**")
            source_ids = list(item.evidence.get("metrics", []))
            if comparison:
                source_ids += comparison.metric_ids + comparison.normalization_sources
            for source_id in dict.fromkeys(source_ids):
                if source_id in values:
                    _show_value(values[source_id])


def _render_hypotheses(run, hypotheses, events, ingestion, prefix, synthetic):
    with st.expander("Добавить прогноз для будущей проверки"):
        with st.form(prefix + "_forecast"):
            prediction = st.text_input("Прогноз")
            metric = st.selectbox("Показатель прогноза", sorted(set(METRIC_CATALOG) | {value.metric for value in ingestion.values}), format_func=_metric_name)
            direction = st.selectbox("Ожидаемое направление", ["up", "down"], format_func=lambda key: "Рост" if key == "up" else "Снижение")
            check_after = st.date_input("Проверить после")
            scope = st.text_input("Подразделение прогноза")
            event_ids = [""] + [event.id for event in events]
            event_id = st.selectbox("Связанное событие", event_ids, format_func=lambda key: key or "Не выбрано")
            submit = st.form_submit_button("Сохранить прогноз")
        if submit:
            if not prediction.strip() or not scope.strip() or not event_id:
                st.error("Укажите прогноз, подразделение и событие с источником.")
            else:
                hypotheses.append(Hypothesis(id="hyp_manual_" + uuid4().hex[:10], source_type="manual", source_id=event_id, prediction=prediction.strip(), metric=metric, expected_direction=direction, check_after=check_after.isoformat(), unit_scope=scope.strip(), event_id=event_id, is_synthetic=synthetic))
                _invalidate(prefix)
                st.success("Ручной прогноз сохранён. Он будет отмечен отдельно от прогнозов агента; пересчитайте результат.")
    items = run.hypotheses if run else hypotheses
    agent_items = [item for item in items if item.source_type != "manual"]
    confirmed = sum(item.status == "confirmed" for item in agent_items)
    checked = sum(item.status in {"confirmed", "refuted"} for item in agent_items)
    st.metric(data_label(synthetic) + " · Точность проверенных прогнозов агента", f"{confirmed / checked:.0%}" if checked else "Нет проверок")
    st.caption(f"Подтвердилось {confirmed} из {checked} определённых результатов. Ожидающие, неопределённые и ручные прогнозы исключены из знаменателя.")
    if not items:
        st.info("Прогнозов для проверки ещё нет.")
    for item in items:
        with st.container(border=True):
            st.caption(data_label(item.is_synthetic) + " · " + HYPOTHESES.get(item.status, item.status))
            st.write(item.prediction)
            st.caption(f"{_metric_name(item.metric)} · {item.unit_scope} · проверить после {item.check_after} · источник: {item.source_type}")
            st.write(item.explanation)
            st.caption("Сравнения: " + (", ".join(item.checked_with) or "ещё не выполнены"))


def _render_recommendations(run):
    if not run.recommendations:
        st.info("Обоснованных рекомендаций пока нет. Подтвердите механизм связи и накопите сопоставимые наблюдения.")
    for item in run.recommendations:
        with st.container(border=True):
            st.caption(data_label(item.is_synthetic) + " · Уверенность: " + CONFIDENCE.get(item.confidence, item.confidence))
            st.write(item.action)
            st.text("Ожидаемый эффект: " + json.dumps(item.expected_effect, ensure_ascii=False))
            st.text("Основания: " + json.dumps(item.based_on, ensure_ascii=False))
            for risk in item.risks:
                st.write("Риск: " + risk)
            if item.redline_id:
                st.caption("Связанная правка текста: " + item.redline_id)
            st.caption("Решение: " + {"accept": "принято", "reject": "отклонено", "discuss": "обсудить"}.get(item.decision, item.decision))
            for column, decision, title in zip(st.columns(3), ("accept", "reject", "discuss"), ("Принять", "Отклонить", "Обсудить")):
                if column.button(title, key=run.run_id + "_" + item.id + "_" + decision):
                    try:
                        service.save_decision(run, item.id, decision)
                        item.decision = decision
                        st.rerun()
                    except Exception as exc:
                        _show_error(exc, "Решение не удалось сохранить")
            st.caption("Кнопка сохраняет решение человека по рекомендации. Изменения структуры применяются отдельно.")


def _render_knowledge(run):
    if not run.knowledge:
        st.info("В базе пока нет наблюдений с достаточным основанием.")
    for item in run.knowledge:
        with st.container(border=True):
            st.caption(data_label(item.is_synthetic) + " · " + CONFIDENCE.get(item.confidence, item.confidence))
            st.write(item.change_pattern)
            qualification = "Единичное наблюдение" if item.n_cases < 2 else "Наблюдалось в похожих случаях"
            st.write(f"{qualification} · случаев: {item.n_cases} · разных контекстов: {item.diversity}")
            st.text("Контекст: " + json.dumps(item.context, ensure_ascii=False))
            st.text("Наблюдаемый эффект: " + json.dumps(item.observed_effect, ensure_ascii=False))
            st.caption("Основания: " + ", ".join(item.cases))
            if item.n_cases < 3 or item.diversity < 2:
                st.caption("Данных недостаточно для количественного обещания эффекта.")


def render_metrics_workspace(core_result=None, core_is_synthetic=False):
    st.header("Эффективность и оптимизация")
    st.caption("Структурное изменение → наблюдаемые показатели → проверка прогноза. Связь по времени не доказывает причинность; решение принимает человек.")
    mode = st.radio("Источник показателей", [REAL_LABEL, TEST_LABEL], horizontal=True, key="metrics_data_mode")
    synthetic = mode == TEST_LABEL
    prefix = "metrics_demo" if synthetic else "metrics_real"
    if synthetic:
        st.warning(TEST_LABEL + " · Демонстрационные показатели и события. Это не результаты вашей компании.")
    else:
        st.info("Реальные отчёты хранятся отдельно от синтетического примера. Каждое число связано с исходным файлом и периодом.")
    ingestion = st.session_state.setdefault(prefix + "_ingestion", IngestionResult())
    events = st.session_state.setdefault(prefix + "_events", [])
    hypotheses = st.session_state.setdefault(prefix + "_hypotheses", [])
    controls = st.session_state.setdefault(prefix + "_controls", {})
    run = st.session_state.get(prefix + "_run")
    as_of = st.date_input("Дата проверки прогнозов", key=prefix + "_as_of")
    use_ai = st.checkbox("ИИ для текстовых отчётов и проверки связей", value=False, disabled=synthetic, key=prefix + "_use_ai")
    if use_ai:
        st.caption("По кнопкам обработки фрагменты отчётов, события и показатели будут переданы в OpenAI с настроенным серверным ключом.")
    if synthetic and st.button("Запустить демо эффективности", type="primary"):
        try:
            demo_ingestion, demo_events, demo_hypotheses, demo_controls = service.demo_inputs()
            _same_kind(demo_ingestion, True)
            st.session_state[prefix + "_ingestion"] = demo_ingestion
            st.session_state[prefix + "_events"] = demo_events
            st.session_state[prefix + "_hypotheses"] = demo_hypotheses
            st.session_state[prefix + "_controls"] = demo_controls
            st.session_state[prefix + "_run"] = service.run_metrics(demo_ingestion.values, demo_events, hypotheses=demo_hypotheses, control_scopes=demo_controls, use_ai=False, as_of=as_of.isoformat(), custom_definitions=demo_ingestion.custom_definitions)
            st.rerun()
        except Exception as exc:
            _show_error(exc, "Демо не завершено")
    if st.button("Рассчитать эффективность", disabled=not ingestion.values, type="primary", key=prefix + "_run_button"):
        try:
            _same_kind(ingestion, synthetic)
            with st.spinner("Сопоставляем периоды, проверяем механизмы и прогнозы…"):
                st.session_state[prefix + "_run"] = service.run_metrics(ingestion.values, events, hypotheses=hypotheses, control_scopes=controls, use_ai=use_ai, as_of=as_of.isoformat(), custom_definitions=ingestion.custom_definitions)
            st.rerun()
        except Exception as exc:
            _show_error(exc, "Расчёт не завершён")
    reports_tab, dashboard_tab, effects_tab, hypotheses_tab, recommendations_tab, knowledge_tab = st.tabs(["Отчёты", "Дашборд метрик", "Изменение → эффект", "Прогноз vs факт", "Рекомендации", "База знаний"])
    with reports_tab:
        st.subheader(data_label(synthetic) + " · Источники показателей")
        if not synthetic:
            uploads = st.file_uploader("Отчёты с показателями", type=["xlsx", "xlsm", "pdf", "docx"], accept_multiple_files=True, key="metrics_uploads")
            use_folder = st.checkbox("Добавить отчёты из data/reports", value=False)
            scope = st.text_input("Подразделение для всего комплекта, если известно", placeholder="Оставьте пустым, если в отчётах несколько подразделений")
            if st.button("Извлечь показатели", key="metrics_ingest"):
                files = [(uploaded.name, uploaded.getvalue()) for uploaded in uploads]
                if use_folder and REPORTS_DIR.exists():
                    files.extend((path.name, path.read_bytes()) for path in sorted(REPORTS_DIR.iterdir()) if path.is_file() and path.suffix.lower() in {".xlsx", ".xlsm", ".pdf", ".docx"})
                if not files:
                    st.error("Добавьте отчёты или поместите их в data/reports.")
                elif any(name.upper().startswith("SYNTH_") for name, _ in files):
                    st.error("Файлы SYNTH_ относятся к тестовому режиму и не загружаются в реальные отчёты.")
                else:
                    try:
                        with st.spinner("Извлекаем показатели и сохраняем ссылки на источники…"):
                            extracted = service.ingest_reports(files, unit_scope=scope.strip() or None, is_synthetic=False, use_ai=use_ai)
                        _same_kind(extracted, False)
                        st.session_state[prefix + "_ingestion"] = extracted
                        _invalidate(prefix)
                        st.rerun()
                    except Exception as exc:
                        _show_error(exc, "Извлечение не завершено")
        for warning in ingestion.warnings:
            st.warning(data_label(synthetic) + " · " + warning)
        if ingestion.values:
            st.dataframe(pd.DataFrame(value_rows(ingestion.values)), hide_index=True, width="stretch")
            lookup = {value.id: value for value in ingestion.values}
            selected = st.selectbox("Проверить число по источнику", list(lookup), format_func=lambda key: data_label(lookup[key].is_synthetic) + " · " + _metric_name(lookup[key].metric) + " · " + lookup[key].period_start)
            _show_value(lookup[selected])
        if ingestion.custom_definitions:
            with st.expander("Подтверждённые пользовательские показатели"):
                st.dataframe(pd.DataFrame([{"Данные": data_label(synthetic), "Код": item.code, "Описание": item.name, "Единица": item.unit, "Подтверждено": item.confirmed_by_human} for item in ingestion.custom_definitions]), hide_index=True, width="stretch")
        _render_queue(ingestion, prefix, synthetic)
        _render_events(events, controls, ingestion, prefix, synthetic, core_result, core_is_synthetic)
    if run:
        if run.warnings:
            with st.expander(f"Ограничения расчёта · {len(run.warnings)}"):
                for warning in run.warnings:
                    st.warning(warning if warning.startswith(data_label(run.is_synthetic)) else data_label(run.is_synthetic) + " · " + warning)
    with dashboard_tab:
        if run:
            _render_dashboard(run)
        else:
            st.info("Загрузите показатели и запустите расчёт. Для автономного примера переключитесь на тестовые данные.")
    with effects_tab:
        if run:
            _render_attributions(run)
        else:
            st.info("Связи появятся после расчёта; для каждой будет показан уровень уверенности и альтернативы.")
    with hypotheses_tab:
        _render_hypotheses(run, hypotheses, events, ingestion, prefix, synthetic)
    with recommendations_tab:
        if run:
            _render_recommendations(run)
        else:
            st.info("Сначала рассчитайте сравнение периодов.")
    with knowledge_tab:
        if run:
            _render_knowledge(run)
        else:
            st.info("База содержит только наблюдения с достаточным основанием и разделяет реальные и тестовые данные.")
    if run:
        st.divider()
        first, second = st.columns(2)
        marker = "SYNTH_" if run.is_synthetic else ""
        first.download_button(data_label(run.is_synthetic) + " · Скачать заключение", service.render_markdown(run), file_name=marker + "metrics-report.md", mime="text/markdown", width="stretch")
        second.download_button(data_label(run.is_synthetic) + " · Скачать данные", json.dumps(run.to_dict(), ensure_ascii=False, indent=2), file_name=marker + "metrics-run.json", mime="application/json", width="stretch")
