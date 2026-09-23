from __future__ import annotations

import html
import json
from collections import Counter

import pandas as pd
import streamlit as st

from qaitu.ai_reviewer import MAX_BATCHES, MAX_BATCH_CHARS, MAX_REVIEW_SECONDS, review_with_llm
from qaitu.analyzer import analyze_documents
from qaitu.demo import demo_documents
from qaitu.extractors import extract_document
from qaitu.presentation import apply_theme, result_navigation, sidebar_brand, welcome, workspace_header
from qaitu.reporting import COVERAGE_LABELS, FINDING_LABELS, NORM_LABELS, ROLE_LABELS, STATUS_LABELS, collect_sources, compact_unit_labels, markdown_report, ordered_matrix_rows


st.set_page_config(page_title="QAITU · карта ответственности", page_icon="◈", layout="wide")
apply_theme()


def render_source(source, *, context=False):
    st.caption(("Контекст · " if context else "") + ("ДО" if source.period == "before" else "ПОСЛЕ") + " · " + source.document)
    st.caption(f"{source.locator} · {source.id}")
    st.text(source.text)


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


def render_matrix(rows, units, before_units, after_units, selected_id, compact=True):
    # Every document-derived value is escaped before entering this HTML table.
    esc = lambda value: html.escape(str(value), quote=True)
    count = len(units)
    unit_labels = compact_unit_labels(units) if compact else {unit: unit for unit in units}
    table_class = "qa-matrix qa-compact" if compact else "qa-matrix"
    parts = [f'<div class="qa-matrix-wrap" role="region" aria-label="Матрица функций до и после" tabindex="0"><table class="{table_class}"><thead>',
             f'<tr><th rowspan="2" class="qa-label">Функция / назначение</th><th colspan="{count}" class="qa-group">ДО</th><th colspan="{count}" class="qa-group qa-divider">ПОСЛЕ</th></tr><tr>']
    for side in ("before", "after"):
        for index, unit in enumerate(units):
            divider = "qa-divider" if side == "after" and index == 0 else ""
            parts.append(f'<th class="{divider}" title="{esc(unit)}">{esc(unit_labels[unit])}</th>')
    parts.append("</tr></thead><tbody>")
    for row in rows:
        label = row.label if len(row.label) <= 170 else row.label[:167] + "…"
        note = STATUS_LABELS.get(row.status, row.status) + (" · ⚑ проверить пересечение" if row.candidate_overlap else "")
        selected = "qa-selected" if row.id == selected_id else ""
        parts.append(f'<tr class="{selected}"><td class="qa-label" title="{esc(row.label)}"><span class="qa-function-text">{esc(label)}</span><small>{esc(note)}</small></td>')
        for side, present in (("before", before_units), ("after", after_units)):
            for index, unit in enumerate(units):
                css, text, description = matrix_cell(row, unit, side, present)
                divider = " qa-divider" if side == "after" and index == 0 else ""
                parts.append(f'<td class="{css}{divider}" title="{esc(description)}">{esc(text)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


with st.sidebar:
    sidebar_brand()
    st.divider()
    st.markdown('<div class="qa-sidebar-title">Комплекты документов</div><div class="qa-sidebar-hint">Добавьте исходную и новую редакции.</div>', unsafe_allow_html=True)
    before_files = st.file_uploader("До изменений", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True)
    after_files = st.file_uploader("После изменений", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True)
    run = st.button("Сравнить документы", type="primary", width="stretch")
    demo = st.button("Запустить контрольный пример", width="stretch")
    with st.expander("Дополнительная ИИ-проверка"):
        use_llm = st.checkbox("Разрешаю отправку фрагментов во внешний API", value=False)
        st.caption("Опциональная проверка через OpenAI. Включайте только для данных, которые разрешено передавать этому сервису. Основной анализ работает локально.")
        st.caption(f"До {MAX_BATCHES} пакетов по {MAX_BATCH_CHARS:,} символов; бюджет до {MAX_REVIEW_SECONDS:g} секунд. Проверка может охватить только часть комплекта — ограничения появятся в результате.".replace(",", " "))
        api_key = st.text_input("OpenAI API key", type="password", disabled=not use_llm)
        model = st.text_input("Модель", value="gpt-4.1-mini", disabled=not use_llm)
        st.caption("Ключ не включается в отчёты и не записывается приложением в файлы.")
    st.divider()
    st.caption("PDF с текстом · DOCX · XLSX / XLSM")
    st.caption("Для сканированных PDF нужен предварительный OCR. Выводы требуют проверки сотрудником.")

workspace_header()
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
                local_result = analyze_documents(before_docs, after_docs)
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
if result is None:
    if welcome():
        st.session_state["run_welcome_demo"] = True
        st.rerun()
    st.stop()

result_navigation()
st.header("Обзор анализа", anchor="overview")
sources = collect_sources(result)
document_packs = st.session_state.get("document_packs", {})
is_demo = st.session_state.get("mode") == "demo"
if is_demo:
    st.info("КОНТРОЛЬНЫЙ ПРИМЕР · Синтетические документы для проверки сценария. Эти результаты не относятся к вашим положениям.")
else:
    st.caption("ЗАГРУЖЕННЫЕ ДОКУМЕНТЫ · Результат последнего завершённого сравнения")
st.caption(st.session_state.get("llm_status", "Локальный анализ"))
if st.session_state.get("llm_error"):
    st.warning(st.session_state["llm_error"])
if result.warnings:
    with st.expander(f"Ограничения обработки · {len(result.warnings)}", expanded=False):
        for warning in result.warnings:
            st.warning(warning)

st.caption(
    f"Охват: документов до/после — {result.coverage.get('documents_before', 0)}/{result.coverage.get('documents_after', 0)}; "
    f"исходных фрагментов — {result.coverage.get('fragments_before', 0) + result.coverage.get('fragments_after', 0)}."
)
counts = Counter(finding.kind for finding in result.findings)
columns = st.columns(4)
for column, label, number, explanation in zip(columns,
        ["Изменения владельцев", "Возможные потери", "Возможные дубли", "Конфликты ролей"],
        [sum(row.status == "moved" for row in result.matrix_rows), counts["loss"], counts["duplicate"], counts["conflict"]],
        ["Строки с передачей или изменением набора владельцев", "Закрепление не найдено в обработанном комплекте", "Кандидаты для проверки ролей и области", "Индикаторы исполнения и проверки одного процесса"]):
    column.metric(label, number, help=explanation)
st.caption("Количество индикаторов, а не подтверждённых нарушений. Числовая уверенность эвристики не является вероятностью правильного вывода.")

summary_section = st.container(key="summary_section")
units_section = st.container(key="units_section")
matrix_section = st.container(key="matrix_section")
sources_section = st.container(key="sources_section")

with matrix_section:
    st.header("Функция × подразделение", anchor="matrix")
    st.caption("Одинаковые колонки до / после. Статусы рассчитаны по полному комплекту; фильтры меняют только отображение.")
    pack_columns = st.columns(2)
    for column, period, label in zip(pack_columns, ("before", "after"), ("ДО", "ПОСЛЕ")):
        pack = document_packs.get(period, [])
        actual_names = {source.document for source in sources if source.period == period}
        if not pack or {item["name"] for item in pack} != actual_names:
            pack = [{"name": name, "metadata": {}} for name in sorted(actual_names)]
        if len(pack) == 1:
            metadata = pack[0]["metadata"]
            details = [f"Редакция {metadata['edition']}" if metadata.get("edition") else pack[0]["name"]]
            if metadata.get("protocol"):
                details.append(f"протокол №{metadata['protocol']}")
            if metadata.get("date"):
                details.append(metadata["date"])
            column.caption(label + " · " + " · ".join(details))
        else:
            column.caption(f"{label} · документов: {len(pack)}")
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
    st.caption(f"Показано {len(rows)} из {len(result.matrix_rows)} строк · {len(units)} владельцев, включая общие роли · запреты доступны в источниках и контексте")
    if not rows or not units:
        st.info("Для выбранных условий строки не найдены. Измените фильтр или проверьте охват извлечения.")
    else:
        page_size = 10
        page_count = (len(rows) + page_size - 1) // page_size
        display_options, row_picker = st.columns([1, 2.5], vertical_alignment="bottom")
        with display_options:
            compact = st.checkbox("Компактные названия колонок", value=True)
            page = st.selectbox("Страница матрицы", list(range(page_count)), format_func=lambda value: f"{value + 1} / {page_count}") if page_count > 1 else 0
        page_rows = rows[page * page_size:(page + 1) * page_size]
        lookup = {row.id: row for row in page_rows}
        with row_picker:
            selected_id = st.selectbox("Выберите строку для просмотра цитат", list(lookup), format_func=lambda key: lookup[key].label[:150], key="matrix_row_selection")
        render_matrix(page_rows, units, before_units, after_units, selected_id, compact=compact)
        if compact:
            with st.expander("Обозначения владельцев · полные названия"):
                st.caption("П1, П2… — обозначения колонок этой матрицы. Наведите указатель на заголовок, чтобы увидеть полное название.")
                st.dataframe(pd.DataFrame([{"Колонка": label, "Владелец": unit} for unit, label in compact_unit_labels(units).items()]), hide_index=True, width="stretch")
        st.caption("И — исполнение · У — участие · С — согласование · Т — утверждение · К — контроль · Ко — координация")
        st.caption("Синий + / − — изменение назначения · красный − — закрепление не найдено · ⚑ — проверить пересечение · ? — недостаточно данных · × — нет в перечне редакции · · — нет прямого закрепления")
        row = lookup[selected_id]
        st.divider()
        st.subheader("Основание выбранной строки")
        st.write(row.label)
        st.caption(NORM_LABELS.get(row.norm_type, row.norm_type) + " · " + STATUS_LABELS.get(row.status, row.status))
        if row.candidate_overlap:
            st.warning("Несколько владельцев: сопоставьте роли и область ответственности. Несколько отметок сами по себе не доказывают дубль.")
        for note in row.notes:
            st.info(note)
        with st.container(key="evidence_columns"):
            before_column, after_column = st.columns(2, gap="medium")
        with before_column:
            st.markdown("**ДО / исходные назначения**")
            render_assignments(row.before)
        with after_column:
            st.markdown("**ПОСЛЕ / найденные назначения**")
            render_assignments(row.after)

with summary_section:
    st.header("Что проверить в первую очередь", anchor="conclusion")
    st.caption("Каждый вывод опирается на фрагменты. Подтверждение нарушения и решение о перераспределении остаются за сотрудником.")
    unit_counts = Counter(change.status for change in result.unit_changes)
    assignment_counts = Counter(row.status for row in result.matrix_rows)
    st.write(
        f"Подразделения: сохранено {unit_counts['preserved']}, возможно преобразовано {unit_counts['transformed']}, "
        f"появилось в перечне {unit_counts['created']}, не найдено в новом перечне {unit_counts['removed']}. "
        f"Назначения: сохранено {assignment_counts['preserved']}, передано/изменены владельцы {assignment_counts['moved']}, "
        f"изменена формулировка {assignment_counts['changed']}, без прямого соответствия до {assignment_counts['new']}."
    )
    st.caption("«Без прямого соответствия до» не означает, что функция впервые появилась в компании. Сильные переформулировки и неполные комплекты требуют экспертной проверки.")
    if not result.findings:
        st.info("Индикаторы рисков не найдены. Проверьте охват и назначения: это не подтверждение отсутствия рисков.")
    kinds = st.multiselect("Тип проверки", ["loss", "duplicate", "conflict"], format_func=lambda key: FINDING_LABELS[key], placeholder="Все типы")
    for index, finding in enumerate(result.findings, 1):
        if kinds and finding.kind not in kinds:
            continue
        with st.expander(f"{index:02d} · {FINDING_LABELS.get(finding.kind, finding.kind)} · {finding.title}", expanded=False):
            st.write(finding.explanation)
            st.markdown("**Рекомендация**")
            st.write(finding.recommendation or "Сверить назначение и границы ответственности с владельцем процесса.")
            st.caption("Пункты, на которых основан индикатор")
            for source in finding.sources:
                render_source(source)
                st.divider()
    st.markdown("**Изменения структуры**")
    for change in result.unit_changes:
        st.write(f"{STATUS_LABELS.get(change.status, change.status)}: {change.before or '—'} → {change.after or '—'}")
    st.caption("Полная матрица доступна в разделе «Матрица функций» и в скачиваемом подробном отчёте.")

with units_section:
    st.header("Изменения организационной структуры", anchor="structure")
    st.caption("Появление или отсутствие в перечне не доказывает юридическое создание, ликвидацию или преобразование без распорядительного документа.")
    if result.unit_changes:
        data = [{"Статус": STATUS_LABELS.get(item.status, item.status), "До": item.before or "—", "После": item.after or "—", "Источники": ", ".join(source.id for source in item.sources)} for item in result.unit_changes]
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True, row_height=56,
            height=min(56 * len(data) + 40, 440),
            column_config={"Статус": st.column_config.TextColumn(width="medium"),
                           "До": st.column_config.TextColumn(width="large"),
                           "После": st.column_config.TextColumn(width="large"),
                           "Источники": st.column_config.TextColumn(width="medium")})
        unit_index = st.selectbox("Проверить изменение по источникам", range(len(result.unit_changes)), format_func=lambda index: (result.unit_changes[index].before or "—") + " → " + (result.unit_changes[index].after or "—"))
        for source in result.unit_changes[unit_index].sources:
            with st.container(border=True):
                render_source(source)
    else:
        st.info("Изменения структуры не выделены. Проверьте заголовки и качество извлечённого текста.")

with sources_section:
    st.header("Проверяемость и охват", anchor="sources")
    source_counts = Counter(source.period for source in sources)
    c1, c2, c3 = st.columns(3)
    c1.metric("Фрагменты до", source_counts["before"])
    c2.metric("Фрагменты после", source_counts["after"])
    c3.metric("Строки функций", len(result.matrix_rows))
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
    st.caption("Каталог включает исходные фрагменты, обе стороны сопоставлений и контекст владельцев. Здесь можно проверить и нормативные запреты.")
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
st.header("Скачать результат", anchor="export")
download1, download2, note = st.columns([1, 1, 2])
with download1:
    st.download_button("↓ Заключение Markdown", markdown_report(result, is_demo=is_demo), file_name="qaitu-conclusion.md", mime="text/markdown", width="stretch", on_click="ignore")
with download2:
    st.download_button("↓ Полный результат JSON", json.dumps(result.to_dict(), ensure_ascii=False, indent=2), file_name="qaitu-analysis.json", mime="application/json", width="stretch", on_click="ignore")
with note:
    st.caption("Экспорт включает полный анализ и источники. Активные фильтры не сокращают отчёт.")
