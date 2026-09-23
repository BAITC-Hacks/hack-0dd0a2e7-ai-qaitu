"""Designed, offline PDF reports. No Markdown conversion or remote resources."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, LongTable, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle

from .confidence import confidence_counts, metric_lines
from .exports import DISCLAIMER, clean_text, clip, document_lines, executive_conclusion, level_label, overview, score_of
from .models import AnalysisResult
from .reporting import AI_COMPARISON_LABELS, AI_STATUS_LABELS, FINDING_LABELS, NORM_LABELS, ROLE_LABELS, STATUS_LABELS, ai_review_overview, collect_sources, compact_unit_labels, ordered_matrix_rows


FONT_LOCK = Lock()
PDF_BUILD_LOCK = Lock()
FONT_DIR = Path(__file__).with_name("static") / "fonts"
NAVY = colors.HexColor("#17364A")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#596B79")
LIGHT = colors.HexColor("#F0F5F8")
BAND_COLORS = {"very_high": "#B33835", "high": "#AD6200", "review": "#8A7400", "weak": "#66727D"}
MARGIN = 17 * mm
SHORT_RISKS = 10
SHORT_CHANGES = 8
SHORT_AI_COMPARISONS = 5


def _register_fonts():
    with FONT_LOCK:
        if "Qaitu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Qaitu", str(FONT_DIR / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("QaituBold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
            pdfmetrics.registerFontFamily("Qaitu", normal="Qaitu", bold="QaituBold", italic="Qaitu", boldItalic="QaituBold")


class _ReportDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        bookmark = getattr(flowable, "bookmark", None)
        if bookmark:
            self.canv.bookmarkPage(bookmark)
            self.canv.addOutlineEntry(flowable.getPlainText(), bookmark, level=0)


class _PagedTable(LongTable):
    def split(self, availWidth, availHeight):
        parts = super().split(availWidth, availHeight)
        # A remainder may otherwise start in the small space left in this same
        # frame, repeating the header mid-page and leaving a sliver of a row.
        return [parts[0], PageBreak(), *parts[1:]] if len(parts) > 1 else parts


def _build_pdf(result: AnalysisResult, *, full: bool = False, document_packs: dict | None = None, is_demo: bool = False) -> bytes:
    _register_fonts()
    stream = BytesIO()
    kind = "Полный отчёт" if full else "Краткий управленческий отчёт"
    doc = _ReportDoc(stream, title=f"QAITU — {kind}", author="QAITU", pagesize=A4,
                     leftMargin=MARGIN, rightMargin=MARGIN, topMargin=22 * mm, bottomMargin=20 * mm,
                     pageCompression=1)

    def page_decoration(canvas, document):
        canvas.saveState()
        width, height = canvas._pagesize
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1)
        canvas.line(MARGIN, height - 15 * mm, width - MARGIN, height - 15 * mm)
        canvas.setFont("QaituBold", 9)
        canvas.setFillColor(NAVY)
        canvas.drawString(MARGIN, height - 12 * mm, "QAITU / АНАЛИЗ ОРГАНИЗАЦИОННЫХ ИЗМЕНЕНИЙ")
        canvas.setFont("Qaitu", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 12 * mm, kind + " · Кандидаты для экспертной проверки")
        canvas.drawRightString(width - MARGIN, 12 * mm, f"Страница {document.page}")
        canvas.restoreState()

    for name, size in (("portrait", A4), ("landscape", landscape(A4))):
        frame = Frame(MARGIN, 20 * mm, size[0] - 2 * MARGIN, size[1] - 42 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id=name, frames=frame, pagesize=size, onPage=page_decoration))
    styles = {
        "body": ParagraphStyle("body", fontName="Qaitu", fontSize=9, leading=12.3, spaceAfter=6, textColor=NAVY, splitLongWords=True),
        "small": ParagraphStyle("small", fontName="Qaitu", fontSize=7.5, leading=10, spaceAfter=4, textColor=MUTED, splitLongWords=True),
        "cell": ParagraphStyle("cell", fontName="Qaitu", fontSize=8, leading=10.5, spaceAfter=0, textColor=NAVY, splitLongWords=True),
        "head": ParagraphStyle("head", fontName="QaituBold", fontSize=20, leading=25, spaceAfter=14, textColor=NAVY, keepWithNext=True),
        "sub": ParagraphStyle("sub", fontName="QaituBold", fontSize=11, leading=14, spaceBefore=7, spaceAfter=7, textColor=TEAL, keepWithNext=True),
        "th": ParagraphStyle("th", fontName="QaituBold", fontSize=8, leading=11, textColor=colors.white, splitLongWords=True),
    }
    styles["source_meta"] = ParagraphStyle("source_meta", parent=styles["small"], keepWithNext=True)
    story = []
    portrait_width = A4[0] - 2 * MARGIN
    landscape_width = landscape(A4)[0] - 2 * MARGIN
    sources = collect_sources(result)
    source_keys = {s.id: "src-" + sha256(s.id.encode()).hexdigest() for s in sources}
    units = list(dict.fromkeys(f.unit for row in result.matrix_rows for f in row.before + row.after))
    unit_labels = compact_unit_labels(units)

    def p(text, style="body", *, raw=False):
        markup = text if raw else escape(clean_text(text)).replace("\n", "<br/>")
        return Paragraph(markup or "—", styles[style])

    def section(title, *, start=True, template=None):
        if start:
            if template:
                story.append(NextPageTemplate(template))
            story.append(PageBreak())
        heading = p(title, "head")
        heading.bookmark = "section-" + str(len(story))
        story.append(heading)

    def table(headers, rows, proportions, width=portrait_width):
        data = [[p(header, "th") for header in headers]]
        data.extend([[cell if isinstance(cell, Paragraph) else p(cell, "cell") for cell in row] for row in rows])
        if len(data) == 1:
            story.append(p("Нет записей в обработанном комплекте.", "small"))
            return
        grid = _PagedTable(data, colWidths=[width * part for part in proportions], repeatRows=1,
                         splitByRow=1, splitInRow=1, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
        ]))
        story.extend([grid, Spacer(1, 8)])

    def ref(source, *, with_quote=False, short=False):
        title = f"{'До' if source.period == 'before' else 'После'} · {source.document} · {source.locator}"
        if short:
            title = f"{'До' if source.period == 'before' else 'После'} · {clip(source.document, 88)} · {source.locator}"
        markup = escape(clean_text(title)) + "<br/>ID: " + escape(source.id)
        if full:
            markup = f'<link href="#{source_keys[source.id]}" color="#156680">{markup}</link>'
        if with_quote:
            markup += "<br/>«" + escape(clip(source.text, 280) if short else clean_text(source.text)) + "»"
        return p(markup, "small", raw=True)

    def owner_names(functions, *, readable=False):
        def label(function):
            short = unit_labels.get(function.unit, function.unit)
            generic = short.startswith("П") and short[1:].strip("′").isdigit()
            return function.unit if readable and generic else short
        return ", ".join(dict.fromkeys(label(f) for f in functions)) or "—"

    def percent(item):
        score = score_of(item)
        return f"{score:.0%}" if score is not None else "—"

    # First page is purpose-built: bounded text, actual counts, no raw matrix.
    section("Анализ реорганизации", start=False)
    story.append(p(kind, "sub"))
    story.append(p("Синтетический контрольный пример" if is_demo else "Сравнение загруженных комплектов «до → после»", "small"))
    docs = document_lines(result, document_packs)
    for line in docs[:4]:
        story.append(p(clip(line, 160), "small"))
    if len(docs) > 4:
        story.append(p(f"Ещё документов: {len(docs) - 4}. Полный состав — в Excel и полном отчёте.", "small"))
    table(["Показатель", "Результат"], overview(result), [.78, .22])
    story.append(p("Общий вывод", "sub"))
    story.append(p(executive_conclusion(result)))
    story.append(p(DISCLAIMER, "small"))
    review = result.ai_review or {}
    story.append(p("Смысловое ИИ-сравнение: " + AI_STATUS_LABELS.get(review.get("status", "skipped"), review.get("status", "skipped")) +
                   (". Покрытие и ограничения — в отдельном разделе." if review else "."), "small"))
    if result.warnings:
        story.append(p(f"Ограничения обработки: {len(result.warnings)}. " + clip(result.warnings[0], 210), "small"))
    story.append(p("Далее: структура → приоритетные риски → изменения функций" +
                   (" → полная матрица и источники." if full else ". Полные назначения и источники — в Excel или полном PDF."), "small"))

    section("Изменения структуры")
    changes = result.unit_changes if full else result.unit_changes[:12]
    story.append(p(f"Показано изменений: {len(changes)} из {len(result.unit_changes)}. Статус описывает перечни документов, а не юридическое создание или ликвидацию.", "small"))
    table(["Статус", "До", "После"], [(STATUS_LABELS.get(c.status, c.status), c.before or "—", c.after or "—") for c in changes], [.24, .38, .38])
    if full:
        for line in docs:
            story.append(p(line, "small"))
        for warning in result.warnings:
            story.append(p("Ограничение: " + warning, "small"))
    else:
        for warning in result.warnings[:3]:
            story.append(p("Ограничение: " + clip(warning, 350), "small"))
        if len(result.warnings) > 3:
            story.append(p("Остальные ограничения — в полном PDF и Excel.", "small"))

    if review:
        section("Смысловое ИИ-сравнение")
        for title, value in ai_review_overview(result):
            # Diagnostic detail remains complete in the full report and workbook.
            if not full and title.startswith("ИИ · причина отклонения"):
                continue
            story.append(p(title.removeprefix("ИИ · ") + ": " + str(value), "small"))
        comparisons = review.get("comparisons", [])
        selected_comparisons = comparisons if full else comparisons[:SHORT_AI_COMPARISONS]
        story.append(p(f"Показано {len(selected_comparisons)} из {len(comparisons)} выбранных сопоставлений. " +
                       ("Все принятые сопоставления и цитаты включены ниже." if full else
                        "Краткий PDF включает до 5 сопоставлений и до 2 цитат на вывод; все сопоставления и цитаты — в полном PDF и Excel."), "small"))
        source_map = {source.id: source for source in sources}
        for index, comparison in enumerate(selected_comparisons, 1):
            story.append(p(f"ИИ {index} · {AI_COMPARISON_LABELS.get(comparison.get('kind'), 'Сопоставление')}", "sub"))
            story.append(p(comparison.get("title", "") if full else clip(comparison.get("title", ""), 200)))
            story.append(p(comparison.get("explanation", "") if full else clip(comparison.get("explanation", ""), 600)))
            evidence_items = comparison.get("evidence", [])
            shown_evidence = list(evidence_items)
            if not full:
                # Prefer one source from each version so short comparisons
                # retain both sides even when the model listed old quotes first.
                shown_evidence = []
                for period in ("before", "after"):
                    match = next((item for item in evidence_items if source_map.get(item.get("source_id"))
                                  and source_map[item["source_id"]].period == period), None)
                    if match:
                        shown_evidence.append(match)
                shown_evidence.extend(item for item in evidence_items if item not in shown_evidence)
                shown_evidence = shown_evidence[:2]
            for evidence in shown_evidence:
                source = source_map.get(evidence.get("source_id"))
                if source:
                    citation = ref(source, short=not full)
                else:
                    citation = p("Источник отсутствует в каталоге: " + evidence.get("source_id", ""), "small")
                citation.keepWithNext = True
                story.append(citation)
                # Show the literal validated excerpt, not an invented summary.
                story.append(p("«" + evidence.get("quote", "") + "»", "small"))
            if not full and len(evidence_items) > 2:
                story.append(p(f"Ещё цитат: {len(evidence_items) - 2}; см. полный PDF или Excel.", "small"))
            recommendation = comparison.get("recommendation", "")
            story.append(p("Рекомендация: " + (recommendation if full else clip(recommendation, 350))))

    section("Приоритетные риски" if not full else "Все кандидаты на проверку")
    eligible = [(i, f) for i, f in enumerate(result.findings, 1) if full or score_of(f) >= .40]
    eligible.sort(key=lambda item: (-score_of(item[1]), item[0]))
    selected = eligible if full else eligible[:SHORT_RISKS]
    story.append(p(f"Показано {len(selected)} из {len(result.findings)} кандидатов. " +
                   ("Включены слабые сигналы." if full else "До 10 кандидатов с уверенностью от 40%, по убыванию оценки. Пропущенные находки остаются в полном PDF и Excel."), "small"))
    for kind_key, bands in confidence_counts(result.findings).items():
        story.append(p(FINDING_LABELS[kind_key] + ": " + "; ".join(f"{label} — {bands[key]}" for key, label in (
            ("very_high", "очень высокая"), ("high", "высокая"), ("review", "требует проверки"), ("weak", "слабый сигнал"))), "small"))
    if not selected:
        story.append(p("Приоритетные кандидаты не найдены. Это не подтверждает отсутствие рисков."))
    row_map = {row.id: row for row in result.matrix_rows}
    for index, (risk_number, finding) in enumerate(selected):
        if not full and index and index % 2 == 0:
            story.append(PageBreak())
        story.append(p(f"Риск R{risk_number} · {FINDING_LABELS[finding.kind]}", "sub"))
        value = finding.assessment
        color = BAND_COLORS[value.level] if value else "#596B79"
        story.append(p(f'<font color="{color}"><b>Уверенность: {percent(finding)} · {escape(level_label(score_of(finding)))}</b></font>', raw=True))
        if value and value.method == "llm-self-report":
            story.append(p("Самооценка модели, ограничена 89%; не вероятность нарушения. " +
                           ("Дополнительная ИИ-проверка смысла пройдена." if value.metrics.get("semantic_verifier_completed") else
                            "Дополнительная ИИ-проверка смысла не отмечена."), "small"))
        row = row_map.get(finding.matrix_row_id)
        if row:
            owners = "Владельцы до: " + owner_names(row.before, readable=True) + ". После: " + owner_names(row.after, readable=True)
            story.append(p(owners if full else clip(owners, 220), "small"))
        story.append(p(finding.title if full else clip(finding.title, 170)))
        story.append(p("Почему отмечено: " + (finding.explanation if full else clip(finding.explanation, 420))))
        direct = list(dict.fromkeys(f.source for f in row.before + row.after)) if row else finding.sources
        for source in direct if full else direct[:2]:
            story.append(ref(source, with_quote=not full, short=not full))
        if not full and len(direct) > 2:
            story.append(p(f"Ещё исходных пунктов: {len(direct) - 2}; см. полный отчёт.", "small"))
        if finding.kind == "loss":
            story.append(p("После: прямое соответствие в обработанном комплекте не найдено. Возможны перенос или другая формулировка.", "small"))
        if value:
            story.append(p(value.priority, "small"))
            if full:
                for text in [*value.reasons, *value.limitations, *metric_lines(value)]:
                    story.append(p(text, "small"))
                for source in value.evidence:
                    story.append(ref(source))
            else:
                metrics = value.metrics
                extra = []
                if "owners_checked" in metrics:
                    extra.append(f"Проверено известных владельцев: {metrics['owners_checked']}")
                if "nearest_text_similarity" in metrics:
                    extra.append(f"Ближайшее текстовое совпадение: {metrics['nearest_text_similarity']:.0%}")
                if extra:
                    story.append(p(". ".join(extra), "small"))
                if value.limitations:
                    story.append(p("Ограничения: " + clip(" ".join(value.limitations), 300), "small"))
        story.append(p("Рекомендация: " + (finding.recommendation if full else clip(finding.recommendation, 270))))
        story.append(Spacer(1, 5))

    for status, title in (("moved", "Передачи ответственности"), ("new", "Новые закрепления функций")):
        section(title)
        rows = [r for r in ordered_matrix_rows(result.matrix_rows) if r.status == status]
        shown = rows if full else rows[:SHORT_CHANGES]
        story.append(p(f"Показано {len(shown)} из {len(rows)} строк. " + ("Приоритет в кратком отчёте — подразделения, затем общие роли. " if not full else "") +
                       ("Новое закрепление не означает появления совершенно новой функции." if status == "new" else "Изменение набора владельцев не всегда означает полный перенос функции."), "small"))
        table(["Функция", "Было", "Стало", "Оценка"], [
            (r.label if full else clip(r.label, 170), owner_names(r.before), owner_names(r.after), percent(r)) for r in shown], [.52, .18, .18, .12])
        used = {f.unit for r in shown for f in r.before + r.after}
        for unit in sorted(used):
            if unit_labels.get(unit) != unit:
                story.append(p(f"{unit_labels.get(unit, unit)} — {unit}", "small"))

    if full:
        section("Приложение А. Полная функциональная матрица", template="landscape")
        story.append(p("Включены все строки и оба набора назначений, независимо от фильтров интерфейса. Ссылки ведут к полным исходным фрагментам приложения Б. Пустая оценка означает, что уверенность не вычислялась.", "small"))

        def assignments(functions):
            if not functions:
                return p("Закрепление не найдено.", "cell")
            parts = []
            for f in functions:
                parts.extend([f"<b>{escape(clean_text(f.unit))}</b> · {escape(ROLE_LABELS.get(f.role, f.role))}", escape(clean_text(f.text))])
                if f.scope:
                    parts.append("Область: " + escape(clean_text(f.scope)))
                if not f.owner_known:
                    parts.append("Владелец требует проверки")
                for s in dict.fromkeys((f.source, *f.context_sources)):
                    parts.append(f'<link href="#{source_keys[s.id]}" color="#156680">{escape(s.id)} · {escape(clean_text(s.locator))}</link>')
            return p("<br/>".join(parts), "cell", raw=True)

        table(["Статус / ID", "Функция / примечания", "До", "После", "Оценка"], [
            (STATUS_LABELS.get(row.status, row.status) + "\n" + NORM_LABELS.get(row.norm_type, row.norm_type) + "\n" + row.id,
             row.label + ("\n" + "\n".join(row.notes) if row.notes else ""), assignments(row.before), assignments(row.after), percent(row))
            for row in result.matrix_rows], [.11, .20, .31, .31, .07], width=landscape_width)
        section("Приложение Б. Полный каталог источников", template="portrait")
        story.append(p(f"Исходных фрагментов: {len(sources)}. Цитаты сохранены полностью. Управляющие символы, недопустимые для печати, заменены символом замены.", "small"))
        for source in sources:
            story.append(p(f'<a name="{source_keys[source.id]}"/>' + escape(source.id), "sub", raw=True))
            story.append(p(f"{'До' if source.period == 'before' else 'После'} · {source.document} · {source.locator}", "source_meta"))
            story.append(p(source.text))
        story.append(p("Конец полного каталога источников.", "small"))
    doc.build(story)
    return stream.getvalue()


def pdf_report(result: AnalysisResult, *, full: bool = False, document_packs: dict | None = None, is_demo: bool = False) -> bytes:
    # Streamlit deferred downloads run in separate threads. ReportLab's shared
    # TTFont instances must not be used to build two documents simultaneously.
    # No document bytes or user data are stored in shared state.
    with PDF_BUILD_LOCK:
        return _build_pdf(result, full=full, document_packs=document_packs, is_demo=is_demo)
