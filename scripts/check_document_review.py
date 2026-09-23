#!/usr/bin/env python3
"""Acceptance on the actual editions, without copying documents into the repo.

The FEATURES H1 expectation for edition 9 conflicts with the supplied wording:
5.10.2/5.10.5 reference real prohibitions 5.8.1/5.8.2, not nonexistent clauses.
This check deliberately does not manufacture that expected false positive.
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document as WordDocument
from pypdf import PdfReader

from qaitu.analyzer import analyze_documents
from qaitu.approval_export import approval_docx, approval_pdf
from qaitu.extractors import extract_document
from qaitu.linter import LINTER_VERSION, inspect_document_pack, lint_document
from qaitu.review import ReviewStore, gate_status, package_id, review_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional local directory for generated approval PDF/DOCX")
    args = parser.parse_args()
    documents = []
    for path, period in ((args.before, "before"), (args.after, "after")):
        with path.open("rb") as handle:
            documents.append(extract_document(handle, path.name, period))
    old, new = documents
    old_lint, new_lint = lint_document(old), lint_document(new)
    result = analyze_documents([old], [new])
    result.document_checks = old_lint + new_lint
    result.analysis_context["linter_version"] = LINTER_VERSION
    checks = []

    def check(label, ok):
        checks.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {label}")

    check("H2: empty 5.5.3 in edition 8, with original quote", any(f.code == "LNT-EMPTY" and f.sources[0].text == "5.5.3. ;" for f in old_lint))
    check("H1 correction: no invented invalid reference in edition 9", not any(
        f.code in {"LNT-REF", "LNT-REF-SEM"} and any(s.text.startswith(("5.10.2.", "5.10.5.")) for s in f.sources) for f in new_lint))
    check("Edition 8: two semantic reference candidates, both with target evidence", sum(f.code == "LNT-REF-SEM" and len(f.sources) >= 3 for f in old_lint) == 2)
    known = {s.id: s for d in documents for s in d.fragments}
    check("Every linter conclusion has exact original evidence", all(f.sources and all(known[s.id] == s and s.text.strip() for s in f.sources) for f in result.document_checks))
    check("Original comparison: three loss candidates remain", sum(f.kind == "loss" for f in result.findings) == 3)
    check("Original comparison: one potential overlap remains", sum(f.kind == "duplicate" for f in result.findings) == 1)
    check("Old hygiene is not a blocker for the new edition", all(f not in old_lint for f in review_items(result)))
    check("Untouched real case: red gate", gate_status(result, [])["color"] == "red")
    single = inspect_document_pack([new])
    check("Single-document mode does not invent a comparison", not single.matrix_rows and not single.function_matches and not single.unit_changes)
    with tempfile.TemporaryDirectory(prefix="qaitu-review-check-") as tmp:
        path = Path(tmp) / "review.sqlite3"
        store = ReviewStore(path)
        package = package_id(result)
        first = review_items(result)[0]
        rejected = False
        try:
            store.save(package, first, status="intentional", actor="Acceptance test", comment="")
        except ValueError:
            rejected = True
        check("Closing without a comment is rejected", rejected and not store.history(package))
        event = store.save(package, first, status="found", actor="Acceptance test", comment="Синтетическое решение для проверки экспорта; вопрос остаётся открытым.")
        history = ReviewStore(path).history(package)
        check("Decision survives a fresh storage instance", history == [event])
        pdf_bytes, word_bytes = approval_pdf(result, history), approval_docx(result, history)
        pdf = PdfReader(BytesIO(pdf_bytes))
        text = " ".join(page.extract_text() for page in pdf.pages)
        check("Readable PDF with decision and quotes", "Acceptance test" in text and "Источник" in text and "Лист согласования" in text)
        word = WordDocument(BytesIO(word_bytes))
        text = " ".join(p.text for p in word.paragraphs)
        check("Word opens with sources and complete history", "Acceptance test" in text and "История решений" in text and first.sources[0].text in text)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "qaitu-approval.pdf").write_bytes(pdf_bytes)
            (args.output / "qaitu-approval.docx").write_bytes(word_bytes)
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
