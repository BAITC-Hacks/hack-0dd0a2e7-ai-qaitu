from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from qaitu.analyzer import analyze_documents
from qaitu.ai_reviewer import review_with_llm
from qaitu.demo import demo_documents
from qaitu.extractors import extract_document
from qaitu.report import conclusion


st.set_page_config(page_title="QAITU — анализ реорганизации", page_icon="🔎", layout="wide")
st.title("QAITU: анализ организационных изменений")
st.caption("Сравнение структуры и функций «до → после» с прослеживаемостью до источника")


def source_label(source) -> str:
    return f"{source.document}, {source.locator} — «{source.text}»"


def add_llm_findings(result, before_docs, after_docs, key: str, model: str) -> None:
    if not key:
        st.warning("LLM-проверка не запущена: укажите API key или настройте OPENAI_API_KEY на сервере.")
        return
    try:
        with st.spinner("Агент выполняет смысловую перепроверку выводов…"):
            reviewed = review_with_llm(before_docs, after_docs, result, key, model)
    except Exception as exc:
        st.warning(f"LLM-проверка не завершилась ({type(exc).__name__}). Локальный анализ сохранён.")
        return
    existing = {
        (finding.kind, frozenset(source.id for source in finding.sources))
        for finding in result.findings
    }
    for finding in reviewed:
        identity = finding.kind, frozenset(source.id for source in finding.sources)
        if identity not in existing:
            result.findings.append(finding)
            existing.add(identity)


with st.sidebar:
    st.header("Документы")
    before_files = st.file_uploader(
        "До реорганизации", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True
    )
    after_files = st.file_uploader(
        "После реорганизации", type=["pdf", "docx", "xlsx", "xlsm"], accept_multiple_files=True
    )
    run = st.button("Анализировать", type="primary", width="stretch")
    demo = st.button("Запустить контрольный пример", width="stretch")
    with st.expander("Смысловая LLM-проверка (опционально)"):
        use_llm = st.checkbox("Включить второй этап")
        api_key_input = st.text_input("OpenAI API key", type="password", disabled=not use_llm)
        api_key = api_key_input or os.getenv("OPENAI_API_KEY", "")
        model = st.text_input("Модель", value="gpt-4.1-mini", disabled=not use_llm)
        st.caption("При включении фрагменты документов будут отправлены во внешний API. Серверный ключ не показывается в браузере.")
    st.divider()
    st.info("Выводы носят рекомендательный характер. Сканированные PDF необходимо предварительно распознать (OCR).")

if run:
    if not before_files or not after_files:
        st.error("Загрузите хотя бы один документ в каждый комплект: «до» и «после».")
        st.stop()
    try:
        before_docs = [extract_document(file, file.name, "before") for file in before_files]
        after_docs = [extract_document(file, file.name, "after") for file in after_files]
        result = analyze_documents(before_docs, after_docs)
        if use_llm:
            add_llm_findings(result, before_docs, after_docs, api_key, model)
        st.session_state["result"] = result
        st.session_state["document_counts"] = (len(before_docs), len(after_docs))
        st.session_state["mode"] = "uploaded"
    except Exception as exc:
        st.exception(exc)
        st.stop()
elif demo:
    before_docs, after_docs = demo_documents()
    result = analyze_documents(before_docs, after_docs)
    if use_llm:
        add_llm_findings(result, before_docs, after_docs, api_key, model)
    st.session_state["result"] = result
    st.session_state["document_counts"] = (len(before_docs), len(after_docs))
    st.session_state["mode"] = "demo"

result = st.session_state.get("result")
if result is None:
    st.markdown(
        """
        ### Как это работает
        1. Загрузите положения, оргструктуры или приложения в PDF, Word или Excel.
        2. Система выделит подразделения и функции и сопоставит версии.
        3. Каждый риск можно раскрыть до исходной формулировки документа.

        Для быстрой проверки нажмите **«Запустить контрольный пример»**.
        """
    )
    st.stop()

for warning in result.warnings:
    st.warning(warning)

losses = sum(f.kind == "loss" for f in result.findings)
duplicates = sum(f.kind == "duplicate" for f in result.findings)
conflicts = sum(f.kind == "conflict" for f in result.findings)
created = sum(c.status == "created" for c in result.unit_changes)
before_units_count = sum(c.before is not None for c in result.unit_changes)
after_units_count = sum(c.after is not None for c in result.unit_changes)
before_functions_count = sum(m.before is not None for m in result.function_matches)
after_functions_count = sum(m.after is not None for m in result.function_matches)
st.caption(
    f"Распознано: подразделений до — {before_units_count}, после — {after_units_count}; "
    f"функций до — {before_functions_count}, после — {after_functions_count}."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Новые подразделения", created)
c2.metric("Возможные потери", losses)
c3.metric("Дублирования", duplicates)
c4.metric("Конфликты ролей", conflicts)

tab_summary, tab_units, tab_functions, tab_sources = st.tabs(
    ["Заключение", "Подразделения", "Функции", "Источники"]
)

labels = {
    "loss": "🔴 Возможная потеря",
    "duplicate": "🟠 Возможное дублирование",
    "conflict": "🟡 Потенциальный конфликт",
}
status_labels = {
    "preserved": "Сохранено", "created": "Создано", "removed": "Исключено", "transformed": "Преобразовано",
    "moved": "Передано", "changed": "Изменено", "lost": "Возможная потеря", "new": "Новая функция",
}

with tab_summary:
    st.subheader("Аналитическое заключение")
    before_count, after_count = st.session_state["document_counts"]
    st.write(conclusion(result, before_count, after_count))
    if not result.findings:
        st.success("По заданным порогам существенных отклонений не найдено.")
    for index, finding in enumerate(result.findings, 1):
        with st.expander(f"{index}. {labels[finding.kind]} — {finding.title}", expanded=True):
            st.write(finding.explanation)
            st.progress(finding.confidence, text=f"Уверенность индикатора: {finding.confidence:.0%}")
            st.markdown("**Подтверждающие фрагменты:**")
            for source in finding.sources:
                st.markdown(f"- `{source.id}` — {source_label(source)}")
            st.markdown(f"**Рекомендация:** {finding.recommendation}")

with tab_units:
    rows = [{
        "Статус": status_labels[item.status],
        "До": item.before or "—",
        "После": item.after or "—",
        "Уверенность": f"{item.confidence:.0%}",
        "Источники": ", ".join(source.id for source in item.sources),
    } for item in result.unit_changes]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_functions:
    rows = [{
        "Статус": status_labels[item.status],
        "Подразделение до": item.before.unit if item.before else "—",
        "Функция до": item.before.text if item.before else "—",
        "Подразделение после": item.after.unit if item.after else "—",
        "Функция после": item.after.text if item.after else "—",
        "Сходство": f"{item.similarity:.0%}",
    } for item in result.function_matches]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_sources:
    unique = {}
    for change in result.unit_changes:
        for source in change.sources:
            unique[source.id] = source
    for finding in result.findings:
        for source in finding.sources:
            unique[source.id] = source
    for source in unique.values():
        st.markdown(f"**`{source.id}` · {source.document} · {source.locator}**  \n{source.text}")

st.download_button(
    "Скачать полный результат JSON",
    data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    file_name="qaitu-analysis.json",
    mime="application/json",
)
