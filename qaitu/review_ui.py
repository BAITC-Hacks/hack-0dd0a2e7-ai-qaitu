"""Small review panels over the existing shared result, without reanalysis."""
from __future__ import annotations

from functools import partial
import json
import sqlite3

import streamlit as st

from .approval_export import approval_docx, approval_pdf
from .linter import IMPLEMENTED_RULES, LIMITATIONS
from .models import AnalysisResult
from .review import (CLOSED, STATUSES, ReviewStore, current_decisions, finding_id,
                     gate_status, package_id, review_items)


def _source(source):
    st.caption(f"{'ДО' if source.period == 'before' else 'ПОСЛЕ'} · {source.document} · {source.locator}")
    st.caption("ID: " + source.id)
    st.text(source.text)


def load_review_history(result):
    try:
        return ReviewStore().history(package_id(result)), True
    except (OSError, sqlite3.Error, ValueError):
        st.error("История решений недоступна. Проверьте права на локальную базу .qaitu/reviews.sqlite3. Сохранение и экспорт согласования отключены; анализ документов доступен.")
        return [], False


def render_gate(result: AnalysisResult, history, *, available=True):
    if not available:
        st.warning("Статус согласования неизвестен: не удалось прочитать историю решений.")
        return
    gate = gate_status(result, history)
    color = gate["color"]
    label = {"red": "🔴", "yellow": "🟡", "green": "🟢"}[color] + " " + gate["label"]
    {"red": st.error, "yellow": st.warning, "green": st.success}[color](label)
    st.caption(f"Чек-лист: открыто {gate['open']} из {gate['total']} · закрыто {gate['closed']}. Не является разрешением на утверждение документа.")
    with st.expander("Основания статуса согласования"):
        for reason in gate["reasons"]:
            st.text(reason)
        st.caption(gate["limitation"])
        if gate["blocker_ids"]:
            st.text("Блокирующие вопросы: " + ", ".join(gate["blocker_ids"]))


def render_linter(result: AnalysisResult):
    st.subheader("Проверка документа", anchor="document-checks")
    st.caption(LIMITATIONS)
    if not result.analysis_context.get("linter_version"):
        st.info("Для этого сохранённого результата линтер ещё не запускался. Повторите анализ.")
        return
    periods = sorted({s.period for s in result.sources})
    period = st.selectbox("Редакция для проверки", periods,
        index=periods.index("after") if "after" in periods else 0,
        format_func=lambda key: "После / проект" if key == "after" else "До / справочно",
        key="lint_period") if periods else "after"
    problems = [f for f in result.document_checks if f.sources and f.sources[0].period == period]
    codes = st.multiselect("Проверки документа", list(IMPLEMENTED_RULES),
                          format_func=lambda c: c + " · " + IMPLEMENTED_RULES[c], key="lint_codes")
    if codes:
        problems = [f for f in problems if f.code in codes]
    st.caption(f"Кандидатов по выбранным условиям: {len(problems)}. Результаты старой редакции не блокируют согласование новой.")
    if not problems:
        st.info("По выбранным локальным правилам кандидаты не найдены. Это не подтверждение корректности всего документа.")
    page_count = max(1, (len(problems) + 4) // 5)
    page = st.selectbox("Страница проверок документа", range(page_count), key="lint_page",
                        format_func=lambda p: f"{p + 1} / {page_count}") if page_count > 1 else 0
    for finding in problems[page * 5:(page + 1) * 5]:
        with st.expander(f"{finding.code} · {finding.title}"):
            st.caption(finding_id(finding))
            st.text(finding.explanation)
            for source in finding.sources:
                _source(source)


def render_checklist(result: AnalysisResult, history, *, available=True):
    st.subheader("Чек-лист согласования", anchor="checklist")
    st.caption("Решения сохраняются локально для этого комплекта. Имя эксперта вводится вручную и не подтверждает личность. В исходных документах ничего не меняется.")
    if not available:
        return
    items = review_items(result)
    current = current_decisions(history)
    package = package_id(result)
    prefix = "review_" + package[:12]
    st.caption("Критические вопросы: потери, конфликты ролей и экспертный приоритет 3. Исправлено — ещё не закрыто: требуется отдельное подтверждение проверки.")
    view, owner = st.columns(2)
    state = view.selectbox("Статус решения", ["all", "open", "closed"],
                           format_func=lambda v: {"all": "Все", "open": "Открытые", "closed": "Закрытые"}[v], key=prefix + "_filter")
    assignees = sorted({d.get("assignee", "") or "Не назначен" for d in current.values()} | {"Не назначен"})
    assignee_filter = owner.selectbox("Ответственный за проверку", ["Все", *assignees], key=prefix + "_owner")
    filtered = []
    for finding in items:
        decision = current.get(finding_id(finding), {})
        closed = decision.get("status") in CLOSED
        if (state == "closed" and not closed) or (state == "open" and closed):
            continue
        if assignee_filter != "Все" and (decision.get("assignee") or "Не назначен") != assignee_filter:
            continue
        filtered.append(finding)
    st.caption(f"Показано {len(filtered)} из {len(items)} вопросов. Экспорт содержит полный чек-лист независимо от фильтров.")
    if filtered:
        lookup = {finding_id(f): f for f in filtered}
        labels = {key: key[:10] + " · " + finding.title for key, finding in lookup.items()}
        selected = st.selectbox("Вопрос для согласования", list(lookup),
            format_func=lambda k: labels[k],
            key=prefix + "_selected")
        finding = lookup[selected]
        previous = current.get(selected, {})
        revision = previous.get("revision", 0)
        st.caption(selected + " · " + STATUSES[previous.get("status", "found")])
        st.text(finding.explanation)
        with st.expander("Источники вопроса"):
            for source in finding.sources:
                _source(source)
        with st.form(prefix + "_" + selected + "_" + str(revision)):
            actor = st.text_input("Кто принимает решение", max_chars=200)
            assignee = st.text_input("Ответственный", value=previous.get("assignee", ""), max_chars=200)
            choices = ["found", "corrected", "intentional", "rejected"]
            if previous.get("status") == "corrected":
                choices.insert(2, "closed")
            status = st.selectbox("Решение эксперта", choices, format_func=lambda k: STATUSES[k])
            comment = st.text_area("Обоснование решения (обязательно)", max_chars=5000)
            correction = st.text_input("Где внесено исправление (файл / редакция и пункт)",
                                        value=previous.get("correction_reference", ""), max_chars=1000)
            impact = st.selectbox("Экспертный приоритет", [1, 2, 3], index=previous.get("impact", 2) - 1,
                                  help="Ручная оценка важности, не автоматический расчёт вероятности или ущерба. Приоритет 3 блокирует согласование, пока вопрос открыт.")
            submitted = st.form_submit_button("Сохранить решение", type="primary")
        if submitted:
            try:
                ReviewStore().save(package, finding, status=status, actor=actor, comment=comment,
                    assignee=assignee, correction_reference=correction, impact=impact, expected_revision=revision)
            except ValueError as exc:
                st.error(str(exc))
            except (OSError, sqlite3.Error):
                st.error("Не удалось сохранить решение. Проверьте доступ к локальной базе; изменение не подтверждено.")
            else:
                st.rerun()
        entries = [event for event in history if event["finding_id"] == selected]
        if entries:
            with st.expander(f"История решений · {len(entries)}"):
                for event in entries:
                    st.text(f"#{event['revision']} · {event['timestamp']} (UTC) · {event['actor']} · {STATUSES[event['status']]}")
                    st.text(event["comment"])
                    if event.get("correction_reference"):
                        st.caption("Исправление: " + event["correction_reference"])
    else:
        st.info("Вопросов по выбранным условиям нет.")
    docx, pdf, journal = st.columns(3)
    with docx:
        st.download_button("Лист согласования · Word", partial(approval_docx, result, history),
            file_name="qaitu-approval.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", on_click="ignore")
    with pdf:
        st.download_button("Лист согласования · PDF", partial(approval_pdf, result, history),
            file_name="qaitu-approval.pdf", mime="application/pdf", on_click="ignore")
    with journal:
        st.download_button("Журнал решений · JSON", json.dumps({"package_id": package, "events": history}, ensure_ascii=False, indent=2),
            file_name="qaitu-review-history.json", mime="application/json", on_click="ignore")
