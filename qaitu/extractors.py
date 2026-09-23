from __future__ import annotations

import io
import hashlib
import re
from pathlib import Path
from typing import BinaryIO, Literal

from .models import Document, Fragment


Period = Literal["before", "after"]


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _fragment(name: str, period: Period, locator: str, text: str, n: int) -> Fragment:
    document_key = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    cleaned = _clean(text)
    clause = re.match(r"^(\d+(?:\.\d+)+)\.", cleaned)
    if clause:
        locator = f"{locator}, п. {clause.group(1)}"
    return Fragment(f"{period}:{document_key}:{n}", name, cleaned, locator, period)


def extract_document(file: BinaryIO, name: str, period: Period) -> Document:
    """Extract traceable text fragments from PDF, DOCX or XLSX."""
    suffix = Path(name).suffix.lower()
    data = file.read()
    fragments: list[Fragment] = []

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        n = 0
        for page_no, page in enumerate(reader.pages, 1):
            for paragraph in re.split(r"\n\s*\n|\n(?=\s*\d+(?:\.\d+)+)", page.extract_text() or ""):
                if _clean(paragraph):
                    n += 1
                    fragments.append(_fragment(name, period, f"стр. {page_no}", paragraph, n))

    elif suffix == ".docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(io.BytesIO(data))
        n = 0
        for p_no, paragraph in enumerate(doc.paragraphs, 1):
            if _clean(paragraph.text):
                n += 1
                fragments.append(_fragment(name, period, f"абзац {p_no}", paragraph.text, n))
        for t_no, table in enumerate(doc.tables, 1):
            for r_no, row in enumerate(table.rows, 1):
                text = " | ".join(_clean(cell.text) for cell in row.cells if _clean(cell.text))
                if text:
                    n += 1
                    fragments.append(_fragment(name, period, f"таблица {t_no}, строка {r_no}", text, n))

    elif suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        n = 0
        for sheet in book.worksheets:
            for r_no, row in enumerate(sheet.iter_rows(values_only=True), 1):
                text = " | ".join(_clean(value) for value in row if _clean(value))
                if text:
                    n += 1
                    fragments.append(_fragment(name, period, f"лист «{sheet.title}», строка {r_no}", text, n))
    else:
        raise ValueError(f"Формат {suffix or 'без расширения'} не поддерживается")

    if not fragments:
        raise ValueError(f"В документе «{name}» не найден извлекаемый текст")
    return Document(name=name, period=period, fragments=fragments)


def document_from_lines(name: str, period: Period, lines: list[str]) -> Document:
    fragments = [
        _fragment(name, period, f"пункт {i}", line, i)
        for i, line in enumerate(lines, 1)
        if _clean(line)
    ]
    return Document(name=name, period=period, fragments=fragments)
