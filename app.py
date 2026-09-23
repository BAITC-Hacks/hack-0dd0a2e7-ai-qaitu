from __future__ import annotations

import html
import json
from collections import Counter
from functools import partial

import pandas as pd
import streamlit as st

from qaitu.ai_reviewer import MAX_BATCHES, MAX_BATCH_CHARS, MAX_REVIEW_SECONDS, review_with_llm
from qaitu.analyzer import analyze_documents
from qaitu.confidence import LEVELS, confidence_counts, confidence_level, metric_lines
from qaitu.demo import demo_documents
from qaitu.extractors import extract_document
from qaitu.exports import excel_report
from qaitu.pdf_report import pdf_report
from qaitu.presentation import apply_theme, result_navigation, sidebar_brand, welcome, workspace_header
from qaitu.reporting import COVERAGE_LABELS, FINDING_LABELS, NORM_LABELS, ROLE_LABELS, STATUS_LABELS, collect_sources, compact_unit_labels, markdown_report, ordered_matrix_rows


st.set_page_config(page_title="QAITU · карта ответственности", page_icon="◈", layout="wide")
apply_theme()


def render_source(source, *, context=False):
    st.caption(("Контекст · " if context else "") + ("ДО" if source.period == "before" else "ПОСЛЕ") + " · " + source.document)
    st.caption(source.locator)
    st.text(source.text)


def render_confidence(value, *, finding=False):
    if value is None:
        st.caption("Подробная оценка отсутствует. Повторите анализ для расчёта уверенности.")
        return
    st.write(f"Уверенность алгоритма: {value.score:.0%} · {LEVELS[value.level]}")
    if finding:
        st.caption(value.priority)
    st.caption("Эвристика, не вероятность нарушения. Уверенность не определяет тяжесть последствий.")
    with st.expander("Почему такая оценка"):
        st.caption("Метод: " + value.method)
        for reason in value.reasons:
            st.text("✓ " + reason)
        for limitation in value.limitations:
            st.text("⚠ " + limitation)
        for line in metric_lines(value):
            st.text(line)
        if value.evidence:
            st.caption("Сравниваемые пункты / ближайшие совпадения — для проверки, не доказательство соответствия")
            for source in value.evidence:
                render_source(source)


def render_assignments(functions):
    if not functions:
        st.info("Закрепление в обработанных документах не найдено. Это не доказывает, что работу никто не выполняет.")
        return
    for function in functions:
        with st.container(border=True):
            st.markdown(f'<h4 class="qa-assignment-title">{html.escape(function.unit)}</h4>', unsafe_allow_html=True)
            st.caption(NORM_LABELS.get(function.norm_type, function.norm_type) + " · " + ROLE_LABELS.get(function.role, function.role))
            if not function.owner_known:
                st.warning("Владелец требует проверки по контексту.")
            if function.scope:
                st.text("Область: " + function.scope)
            st.text(function.text)
            with st.expander("Цитата и контекст назначения", expanded=False):
                render_source(function.source)
                seen = {function.source.id}
                for context in function.context_sources:
                    if context.id not in seen:
                        st.divider()
                        render_source(context, context=True)
                        seen.add(context.id)


ROLE_MARKERS = {"execute": "И", "participate": "У", "approve": "Т", "agree": "С", "coordinate": "Ко", "control": "К", "audit": "К", "review": "К", "unknown": "?"}


def matrix_cell(row, unit, side, units_present):
    assignments = row.before if side == "before" else row.after
    other = row.after if side == "before" else row.before
    current = [function for function in assignments if function.unit == unit]
    previous = [function for function in other if function.unit == unit]
    if current:
        marker = "/".join(sorted({ROLE_MARKERS.get(function.role, "?") for function in current}))
        if any(not function.owner_known for function in current):
            return "qa-unknown", "? " + marker, "Владелец не установлен однозначно"
        if side == "after" and not previous:
            return "qa-change", "+ " + marker, "Добавлено назначение"
        if row.candidate_overlap and side == "after":
            return "qa-overlap", "⚑ " + marker, "Проверить пересечение ролей и областей"
        if side == "after" and row.status == "changed":
            return "qa-change", "~ " + marker, "Формулировка изменилась"
        return "qa-present", marker, "Закрепление найдено"
    if side == "after" and previous:
        if row.status == "unknown":
            return "qa-unknown", "?", "Данных для вывода недостаточно"
        if row.status == "lost" and all(function.owner_known for function in row.before):
            return "qa-missing", "−", "Ответственный не найден в обработанном комплекте"
        return "qa-change", "−", "Прежнее назначение не найдено; проверьте новых владельцев"
    if unit not in units_present:
        return "qa-absent", "×", "В перечне этой редакции не найдено"
    if row.status == "unknown" and not assignments:
        return "qa-unknown", "?", "Данных для вывода недостаточно"
    return "qa-empty", "·", "Прямое закрепление не найдено"


def render_matrix(rows, units, before_units, after_units, selected_id, compact=False):
    # Every document-derived value is escaped before entering this HTML table.
    esc = lambda value: html.escape(str(value), quote=True)
    count = len(units)
    unit_labels = compact_unit_labels(units) if compact else {unit: unit for unit in units}
    table_class = "qa-matrix qa-compact" if compact else "qa-matrix"
    parts = [f'<div class="qa-matrix-wrap" role="region" aria-label="Матрица функций до и после" tabindex="0"><table class="{table_class}"><thead>',
             f'<tr><th rowspan="2" scope="col" class="qa-label">Функция</th><th colspan="{count}" scope="colgroup" class="qa-group">До</th><th colspan="{count}" scope="colgroup" class="qa-group qa-divider">После</th></tr><tr>']
    for side in ("before", "after"):
        for index, unit in enumerate(units):
            divider = "qa-divider" if side == "after" and index == 0 else ""
            parts.append(f'<th scope="col" class="{divider}" title="{esc(unit)}"><span class="qa-unit-name">{esc(unit_labels[unit])}</span></th>')
    parts.append("</tr></thead><tbody>")
    for row in rows:
        label = row.label if len(row.label) <= 170 else row.label[:167] + "…"
        note = {"moved": "Смена владельца", "preserved": "Без изменений", "changed": "Изменена формулировка", "new": "Новое назначение", "lost": "Владелец не найден", "unknown": "Нужна проверка"}.get(row.status, STATUS_LABELS.get(row.status, row.status))
        if row.candidate_overlap:
            note += " · пересечение"
        selected = "qa-selected" if row.id == selected_id else ""
        parts.append(f'<tr class="{selected}"><td class="qa-label" title="{esc(row.label)}"><span class="qa-function-text">{esc(label)}</span><small>{esc(note)}</small></td>')
        for side, present in (("before", before_units), ("after", after_units)):
            for index, unit in enumerate(units):
                css, text, description = matrix_cell(row, unit, side, present)
                divider = " qa-divider" if side == "after" and index == 0 else ""
                if css == "qa-absent":
                    text = "—"
                parts.append(f'<td class="{css}{divider}" title="{esc(description)}" aria-label="{esc(unit + ": " + description + " · " + text)}">{esc(text)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def matrix_legend():
    with st.popover("Как читать", icon=":material/help_outline:"):
        st.markdown("**И** — исполнение · **К** — контроль · **У** — участие · **С** — согласование · **Т** — утверждение · **Ко** — координация")
        st.caption("+ / − — изменение назначения · ⚑ — проверить пересечение · ? — недостаточно данных · — — подразделения нет в перечне редакции · · — нет прямого назначения")
        st.caption("Столбцы до и после расположены в одинаковом порядке. Цвет дополняет обозначения.")


@st.dialog("Матрица функций", width="large")
def expanded_matrix(rows, units, before_units, after_units, selected_id, initial_page=0):
    with st.container(key="expanded_matrix"):
        count, paging, legend = st.columns([3, 1, 1], vertical_alignment="center")
        count.caption(f"Функций: {len(rows)} · владельцев: {len(units)} · текущие фильтры")
        page_size = 25
        page_count = max(1, (len(rows) + page_size - 1) // page_size)
        with paging:
            page = st.selectbox("Страница развёрнутой матрицы", range(page_count),
                index=min(initial_page * 10 // page_size, page_count - 1),
                format_func=lambda value: f"Страница {value + 1} из {page_count}",
                key="expanded_matrix_page", label_visibility="collapsed") if page_count > 1 else 0
        with legend:
            matrix_legend()
        render_matrix(rows[page * page_size:(page + 1) * page_size], units, before_units, after_units, selected_id)


with st.sidebar:
    sidebar_brand()
    st.markdown('<div class="qa-sidebar-title">Документы для сравнения</div>', unsafe_allow_html=True)
    before_files = st.file_uploader("До изменений", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True)
    after_files = st.file_uploader("После изменений", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True)
    after_complete = st.checkbox("Все необходимые документы «после» загружены", value=False,
        help="Ваше подтверждение полноты комплекта, а не результат проверки программы. Если не уверены, оставьте выключенным. Учитывается при следующем анализе.")
    run = st.button("Сравнить документы", type="primary", width="stretch")
    demo = st.button("Запустить контрольный пример", width="stretch")
    with st.expander("Дополнительная ИИ-проверка"):
        use_llm = st.checkbox("Разрешаю отправку фрагментов во внешний API", value=False)
        st.caption("Опциональная проверка через OpenAI. Включайте только для данных, которые разрешено передавать этому сервису. Основной анализ работает локально.")
        st.caption(f"До {MAX_BATCHES} пакетов по {MAX_BATCH_CHARS:,} символов; бюджет до {MAX_REVIEW_SECONDS:g} секунд. Проверка может охватить только часть комплекта — ограничения появятся в результате.".replace(",", " "))
        api_key = st.text_input("OpenAI API key", type="password", disabled=not use_llm)
        model = st.text_input("Модель", value="gpt-4.1-mini", disabled=not use_llm)
        st.caption("Ключ не включается в отчёты и не записывается приложением в файлы.")
    st.caption("PDF с текстом · DOCX · XLSX / XLSM")
    with st.expander("Помощь с документами"):
        st.caption("Для сканов нужен OCR. Добавьте исходную редакцию в «До», новую — в «После».")

demo = demo or st.session_state.pop("run_welcome_demo", False)

if run or demo:
    if run and (not before_files or not after_files):
        st.error("Добавьте хотя бы один документ в каждый комплект: «до» и «после».")
    else:
        try:
            with st.spinner("Читаем документы и сопоставляем назначения…"):
                if demo:
                    before_docs, after_docs = demo_documents()
                else:
                    before_docs = [extract_document(file, file.name, "before") for file in before_files]
                    after_docs = [extract_document(file, file.name, "after") for file in after_files]
                local_result = analyze_documents(before_docs, after_docs, after_complete=after_complete if not demo else False)
            # Save the completed local stage before the optional network stage.
            st.session_state["result"] = local_result
            st.session_state["mode"] = "demo" if demo else "uploaded"
            st.session_state["document_packs"] = {
                period: [{"name": document.name, "metadata": document.metadata, "fragments": len(document.fragments)} for document in documents]
                for period, documents in (("before", before_docs), ("after", after_docs))
            }
            st.session_state["llm_status"] = "Локальный анализ · без передачи документов в API"
            st.session_state["llm_error"] = ""
            if use_llm:
                if not api_key:
                    st.session_state["llm_error"] = "ИИ-проверка пропущена: не указан API key. Локальные результаты сохранены."
                else:
                    try:
                        with st.spinner("Проверяем кандидаты и ссылки дополнительной моделью…"):
                            extra_findings = review_with_llm(before_docs, after_docs, local_result, api_key, model)
                        local_result.findings.extend(extra_findings)
                        st.session_state["llm_status"] = "Локальный анализ + дополнительная ИИ-проверка кандидатов"
                    except Exception as exc:
                        # Provider errors can contain request metadata or secrets.
                        st.session_state["llm_error"] = f"Дополнительная ИИ-проверка не завершилась ({type(exc).__name__}). Локальные результаты доступны; проверьте настройки API и попробуйте снова."
        except Exception as exc:
            st.error(f"Не удалось обработать новый комплект ({type(exc).__name__}). Проверьте формат и наличие извлекаемого текста. Предыдущий результат, если он есть, остаётся ниже.")

result = st.session_state.get("result")
workspace_header()
if result is None:
    if welcome():
        st.session_state["run_welcome_demo"] = True
        st.rerun()
    st.stop()

result_navigation()
st.header("Результат сравнения", anchor="overview")
sources = collect_sources(result)
document_packs = st.session_state.get("document_packs", {})
is_demo = st.session_state.get("mode") == "demo"
if is_demo:
    st.caption("КОНТРОЛЬНЫЙ ПРИМЕР · Синтетические документы, не ваши данные")
else:
    st.caption(f"Документов: {result.coverage.get('documents_before', 0)} до / {result.coverage.get('documents_after', 0)} после · функций в сравнении: {len(result.matrix_rows)}")
if st.session_state.get("llm_error"):
    st.warning(st.session_state["llm_error"])
if result.warnings:
    with st.expander(f"Ограничения обработки · {len(result.warnings)}", expanded=False):
        for warning in result.warnings:
            st.warning(warning)

counts = Counter(finding.kind for finding in result.findings)
with st.container(key="overview_metrics"):
    columns = st.columns(4)
    for column, label, number, explanation in zip(columns,
            ["Смена владельца", "Возможные потери", "Пересечения", "Конфликты ролей"],
            [sum(row.status == "moved" for row in result.matrix_rows), counts["loss"], counts["duplicate"], counts["conflict"]],
            ["Передача или изменение набора владельцев", "Преемник не найден в обработанном комплекте", "Кандидаты на дублирование: нужно сравнить роли и области", "Возможное совмещение исполнения и контроля"]):
        column.metric(label, number, help=explanation)
st.caption("Индикаторы для проверки по цитатам, не подтверждённые нарушения.")

summary_section = st.container(key="summary_section")
matrix_section = st.container(key="matrix_section")
units_section = st.container(key="units_section")
sources_section = st.container(key="sources_section")

with matrix_section:
    st.header("Матрица функций", anchor="matrix")
    f1, f2, f3 = st.columns([1, 1.3, 1.5])
    with f1:
        norm_type = st.selectbox("Вид нормы", ["duty", "right"], format_func=lambda key: {"duty": "Обязанности", "right": "Права"}[key])
    with f2:
        selected_status = st.multiselect("Статус", list(dict.fromkeys(row.status for row in result.matrix_rows)), format_func=lambda key: STATUS_LABELS.get(key, key), placeholder="Все статусы")
    with f3:
        matrix_query = st.text_input("Найти функцию или владельца", placeholder="Например: отчёт, СВК, аудит")
    rows = ordered_matrix_rows([row for row in result.matrix_rows if row.norm_type == norm_type])
    if selected_status:
        rows = [row for row in rows if row.status in selected_status]
    if matrix_query.strip():
        query = matrix_query.casefold().strip()
        rows = [row for row in rows if query in " ".join([row.label, *[function.unit + " " + function.text + " " + function.scope for function in row.before + row.after]]).casefold()]
    before_units = set(result.units_before) | {function.unit for row in result.matrix_rows for function in row.before}
    after_units = set(result.units_after) | {function.unit for row in result.matrix_rows for function in row.after}
    units = sorted(before_units | after_units)
    if not rows or not units:
        st.info("Для выбранных условий строки не найдены. Измените фильтр или проверьте охват извлечения.")
    else:
        page_size = 10
        page_count = (len(rows) + page_size - 1) // page_size
        table_count, table_page, table_help, table_expand = st.columns([4, 1, 1, 1.2], vertical_alignment="center")
        table_count.caption(f"Функций: {len(rows)} из {len(result.matrix_rows)} · владельцев: {len(units)}")
        with table_page:
            page = st.selectbox("Страница матрицы", list(range(page_count)), format_func=lambda value: f"{value + 1} / {page_count}", label_visibility="collapsed") if page_count > 1 else 0
        page_rows = rows[page * page_size:(page + 1) * page_size]
        lookup = {row.id: row for row in page_rows}
        selected_id = st.session_state.get("matrix_row_selection")
        if selected_id not in lookup:
            selected_id = next(iter(lookup))
        with table_help:
            matrix_legend()
        with table_expand:
            if st.button("Развернуть", icon=":material/open_in_full:", key="expand_matrix", width="stretch"):
                expanded_matrix(rows, units, before_units, after_units, selected_id, page)
        render_matrix(page_rows, units, before_units, after_units, selected_id)
        selected_id = st.selectbox("Выберите строку для просмотра цитат", list(lookup), format_func=lambda key: lookup[key].label[:150], key="matrix_row_selection")
        row = lookup[selected_id]
        st.subheader("Сравнить назначения")
        if row.assessment:
            with st.expander(f"Оценка сопоставления · {row.assessment.score:.0%}"):
                render_confidence(row.assessment)
        if row.candidate_overlap:
            st.warning("Несколько владельцев — проверьте роли и границы ответственности.")
        if row.notes:
            with st.expander("Пояснение к сопоставлению"):
                st.caption(STATUS_LABELS.get(row.status, row.status))
                for note in row.notes:
                    st.write(note)
        with st.container(key="evidence_columns"):
            before_column, after_column = st.columns(2, gap="medium")
        with before_column:
            st.markdown("**До**")
            render_assignments(row.before)
        with after_column:
            st.markdown("**После**")
            render_assignments(row.after)

with summary_section:
    review_title, review_filter = st.columns([5, 1], vertical_alignment="center")
    review_title.header("Что проверить", anchor="conclusion")
    with review_filter:
        with st.popover("Фильтр", icon=":material/filter_list:", width="stretch"):
            kinds = st.multiselect("Тип проверки", ["loss", "duplicate", "conflict"], format_func=lambda key: FINDING_LABELS[key], placeholder="Все типы")
            show_weak = st.checkbox("Показать слабые сигналы (менее 40%)", value=False)
            with st.expander("Как читать шкалу уверенности"):
                st.write("90–99% — очень высокая · 70–89% — высокая · 40–69% — требует проверки · 0–39% — слабый сигнал.")
                st.write("Оценка учитывает сходство текста, действий, слов объекта, тип нормы, владельцев и ограничения извлечения. Перенос или разделение снижают уверенность в потере. Это эвристика, не вероятность нарушения.")
                st.caption("Полноту комплекта заявляет пользователь. Отсутствие ошибок чтения не доказывает полноту. 100% и автоматический статус «подтверждено» не выдаются. Приоритет означает порядок проверки, не уровень ущерба.")
                for kind, levels in confidence_counts(result.findings).items():
                    st.write(f"{FINDING_LABELS[kind]}: " + " · ".join(f"{label} — {levels[key]}" for key, label in LEVELS.items()))
    if not result.findings:
        st.info("Индикаторы рисков не найдены. Проверьте охват и назначения: это не подтверждение отсутствия рисков.")
    findings = [finding for finding in result.findings if not kinds or finding.kind in kinds]
    weak_count = sum(confidence_level(finding.confidence) == "weak" for finding in findings)
    if weak_count and not show_weak:
        st.caption(f"Скрыто слабых сигналов: {weak_count}. Они сохранены в общем счётчике и экспорте.")
    if not show_weak:
        findings = [finding for finding in findings if confidence_level(finding.confidence) != "weak"]
    if result.findings:
        st.caption(f"Вопросов: {len(findings)}. Откройте нужный, чтобы увидеть основание и следующий шаг.")
    review_page_count = max(1, (len(findings) + 4) // 5)
    review_page = st.selectbox("Страница вопросов", range(review_page_count), format_func=lambda value: f"{value + 1} / {review_page_count}") if review_page_count > 1 else 0
    for index, finding in enumerate(findings[review_page * 5:(review_page + 1) * 5], review_page * 5 + 1):
        score = finding.assessment.score if finding.assessment else min(.99, finding.confidence)
        with st.expander(f"{index:02d} · {finding.title} · {score:.0%}", expanded=False):
            st.caption(FINDING_LABELS.get(finding.kind, finding.kind))
            render_confidence(finding.assessment, finding=True)
            st.write(finding.explanation)
            st.markdown("**Следующий шаг**")
            st.write(finding.recommendation or "Сверить назначение и границы ответственности с владельцем процесса.")
            st.caption("Пункты, на которых основан индикатор")
            for source in finding.sources:
                render_source(source)
                st.divider()

with units_section:
    st.header("Структура", anchor="structure")
    st.caption("Изменения в загруженных перечнях подразделений.")
    if result.unit_changes:
        data = [{"Статус": STATUS_LABELS.get(item.status, item.status), "До": item.before or "—", "После": item.after or "—"} for item in result.unit_changes]
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True, row_height=56,
            height=min(56 * len(data) + 40, 440),
            column_config={"Статус": st.column_config.TextColumn(width="medium"),
                           "До": st.column_config.TextColumn(width="large"),
                           "После": st.column_config.TextColumn(width="large")})
        with st.expander("Проверить изменение структуры по источникам"):
            st.caption("Изменение перечня само по себе не подтверждает создание или ликвидацию подразделения.")
            unit_index = st.selectbox("Проверить изменение по источникам", range(len(result.unit_changes)), format_func=lambda index: (result.unit_changes[index].before or "—") + " → " + (result.unit_changes[index].after or "—"))
            for source in result.unit_changes[unit_index].sources:
                render_source(source)
    else:
        st.info("Изменения структуры не выделены. Проверьте заголовки и качество извлечённого текста.")

with sources_section:
    st.header("Источники", anchor="sources")
    source_counts = Counter(source.period for source in sources)
    st.caption(f"Фрагментов: {source_counts['before']} до / {source_counts['after']} после")
    with st.expander("Как получен результат"):
        st.caption(st.session_state.get("llm_status", "Локальный анализ"))
        st.write("Сопоставление основано на распознанных фрагментах. Отсутствие назначения не доказывает потерю функции, а новое назначение — появление функции впервые. Проверьте полноту комплекта и существенные переформулировки.")
        st.caption("Оценки эвристики не являются вероятностью правильного вывода. Запреты и контекст владельцев доступны в каталоге ниже.")
    if result.coverage:
        with st.expander("Показатели извлечения"):
            st.caption("Это счётчики обработки, не метрика точности и не гарантия полноты всех обязанностей документа.")
            st.dataframe(pd.DataFrame([{"Показатель": COVERAGE_LABELS.get(key, key), "Количество": value} for key, value in result.coverage.items()]), hide_index=True, width="stretch")
    with st.expander("Состав комплектов и реквизиты документов"):
        for period, label in (("before", "ДО"), ("after", "ПОСЛЕ")):
            for item in document_packs.get(period, []):
                st.text(label + " · " + item["name"])
                metadata = item.get("metadata", {})
                st.caption(" · ".join(f"{key}: {metadata[value]}" for key, value in (("Редакция", "edition"), ("Протокол №", "protocol"), ("Дата", "date")) if metadata.get(value)) or "Реквизиты на титульной странице не распознаны.")
    s1, s2 = st.columns([1, 3])
    with s1:
        source_period = st.selectbox("Редакция", ["all", "before", "after"], format_func=lambda key: {"all": "Обе", "before": "До", "after": "После"}[key])
    with s2:
        source_query = st.text_input("Поиск по тексту, файлу, пункту или ID", placeholder="Например: 5.3.3 или не имеют права")
    filtered_sources = [source for source in sources if source_period == "all" or source.period == source_period]
    if source_query.strip():
        query = source_query.casefold().strip()
        filtered_sources = [source for source in filtered_sources if query in " ".join((source.id, source.document, source.locator, source.text)).casefold()]
    st.caption(f"Найдено фрагментов: {len(filtered_sources)}")
    if filtered_sources:
        source_map = {source.id: source for source in filtered_sources}
        selected_source = st.selectbox("Открыть фрагмент", list(source_map), format_func=lambda key: f"{'До' if source_map[key].period == 'before' else 'После'} · {source_map[key].document} · {source_map[key].locator} · {source_map[key].text[:70]}")
        with st.container(border=True):
            render_source(source_map[selected_source])
    else:
        st.info("Совпадений нет. Попробуйте другой пункт или ключевое слово.")

st.divider()
st.header("Экспорт отчёта", anchor="export")
st.caption("Краткий PDF — сводка, структура, до 10 приоритетных находок и основные изменения. Полный PDF — все кандидаты, матрица и полный каталог источников; может быть большим.")
download_short, download_full, download_excel, download_json = st.columns(4)
# Deferred generation runs only on download, not on every filter change. Bind
# this completed result explicitly; do not access session state from worker threads.
with download_short:
    st.download_button("Краткий PDF", partial(pdf_report, result, full=False, document_packs=document_packs, is_demo=is_demo),
        file_name="qaitu-summary.pdf", mime="application/pdf", icon="📄", width="stretch", on_click="ignore")
with download_full:
    st.download_button("Полный PDF", partial(pdf_report, result, full=True, document_packs=document_packs, is_demo=is_demo),
        file_name="qaitu-full-report.pdf", mime="application/pdf", icon="📑", width="stretch", on_click="ignore")
with download_excel:
    st.download_button("Excel", partial(excel_report, result, document_packs=document_packs, is_demo=is_demo),
        file_name="qaitu-analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon="📊", width="stretch", on_click="ignore")
with download_json:
    st.download_button("JSON", json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        file_name="qaitu-analysis.json", mime="application/json", icon="🧩", width="stretch", on_click="ignore")
st.caption("PDF и Excel формируются локально по нажатию — без API и передачи документов. Фильтры экрана не меняют экспорт. В Excel 8 листов; проценты — числовые, а назначения можно фильтровать по подразделению.")
with st.expander("Дополнительно · технический формат"):
    st.download_button("Markdown", partial(markdown_report, result, is_demo=is_demo),
        file_name="qaitu-conclusion.md", mime="text/markdown", width="stretch", on_click="ignore")
