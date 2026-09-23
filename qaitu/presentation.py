"""Local presentation only; analysis and document data stay in the app layer."""

from pathlib import Path

import streamlit as st


def apply_theme():
    # This CSS approximates glass on the web; it is not Apple's native material.
    st.html(Path(__file__).with_name("static") / "workspace.css")


def sidebar_brand():
    st.markdown('''<div class="qa-brand"><span class="qa-brand-mark" aria-hidden="true">◈</span>
<div><strong>QAITU</strong><span>Рабочее пространство аналитика</span></div></div>''', unsafe_allow_html=True)


def workspace_header():
    st.markdown('''<header class="qa-heading">
<div class="qa-eyebrow">Анализ структуры и функций</div>
<h1>Карта ответственности</h1>
<p>Что изменилось, кому переданы функции и где нужна проверка. С цитатами обеих редакций.</p>
</header>''', unsafe_allow_html=True)


def result_navigation():
    """Native in-page links keep the completed report and filter state intact."""
    with st.container(key="result_navigation"):
        st.markdown('''<nav class="qa-result-nav" aria-label="Разделы результата анализа">
<span class="qa-nav-label">Результат анализа</span>
<div class="qa-nav-links">
<a href="#overview" target="_self">Обзор</a>
<a href="#conclusion" target="_self">Заключение</a>
<a href="#semantic" target="_self">ИИ-сравнение</a>
<a href="#structure" target="_self">Структура</a>
<a href="#matrix" target="_self">Матрица функций</a>
<a href="#sources" target="_self">Источники и охват</a>
<a href="#export" target="_self">Экспорт <span aria-hidden="true">↓</span></a>
</div></nav>''', unsafe_allow_html=True)


def welcome():
    """Return whether the user requested the same local demo as in the sidebar."""
    with st.container(key="welcome"):
        st.markdown('''<section class="qa-welcome">
<div class="qa-welcome-copy"><span class="qa-status">Две редакции. Одна полная картина.</span>
<h2>У каждого изменения<br>есть основание.</h2>
<p>Сопоставьте документы до и после изменений, найдите передачу функций и проверьте ответственных по первоисточнику.</p></div>
<div class="qa-document-scene" role="img" aria-label="Исходная и новая редакции сопоставляются с сохранением источников">
<div class="qa-document qa-document-before"><span class="qa-document-symbol" aria-hidden="true">◈</span><span class="qa-document-label">ДО</span><strong>Исходная<br>редакция</strong><div class="qa-document-lines" aria-hidden="true"><i></i><i></i><i></i></div></div>
<div class="qa-document qa-document-after"><span class="qa-document-symbol" aria-hidden="true">◈</span><span class="qa-document-label">ПОСЛЕ</span><strong>Новая<br>редакция</strong><div class="qa-document-lines" aria-hidden="true"><i></i><i></i><i></i></div></div>
<span class="qa-connection" aria-hidden="true">↔</span></div>
</section>''', unsafe_allow_html=True)
        action, detail = st.columns([1, 1.25], gap="medium")
        with action:
            requested = st.button("Запустить контрольный пример", key="welcome_demo", type="primary", width="stretch", icon=":material/play_arrow:")
        with detail:
            st.caption("Синтетические документы. Без файлов и API-ключа.")
    st.markdown('''<section class="qa-workflow" aria-label="Как работает сравнение">
<div><span class="qa-step-number" aria-hidden="true">1</span><h3>Загрузите две редакции</h3><p>Добавьте файлы в комплекты «До изменений» и «После изменений» в боковой панели.</p></div>
<div><span class="qa-step-number" aria-hidden="true">2</span><h3>Сверьте назначения</h3><p>Проследите передачу функций в матрице. Откройте строку, чтобы увидеть исходные цитаты.</p></div>
<div><span class="qa-step-number" aria-hidden="true">3</span><h3>Передайте заключение</h3><p>Проверьте индикаторы рисков и скачайте полный отчёт с источниками в Markdown или JSON.</p></div>
</section>
<div class="qa-footnote"><span aria-hidden="true">◈</span> Основной анализ выполняется локально. Дополнительная ИИ-проверка включается отдельно.</div>''', unsafe_allow_html=True)
    return requested
