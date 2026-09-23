#!/usr/bin/env python3
"""Explicitly run a paid OpenAI comparison using local server credentials.

Only invoke this command for documents allowed to be sent to OpenAI. The key is
read from OPENAI_API_KEY or the ignored .streamlit/secrets.toml, never arguments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qaitu.ai_agent import run_comparison_agent
from qaitu.analyzer import analyze_documents
from qaitu.config import load_openai_settings
from qaitu.extractors import extract_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, action="append", required=True)
    parser.add_argument("--after", type=Path, action="append", required=True)
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    parser.add_argument("--output", type=Path, help="Save full JSON report to a new private file")
    args = parser.parse_args()
    settings = load_openai_settings()
    if not settings.api_key:
        print("OpenAI key is not configured. Set OPENAI_API_KEY or .streamlit/secrets.toml.", file=sys.stderr)
        return 2
    if args.output and args.output.exists():
        print("Output file already exists; choose a new path.", file=sys.stderr)
        return 2

    def read(paths: list[Path], period: str):
        documents = []
        for path in paths:
            with path.open("rb") as handle:
                documents.append(extract_document(handle, path.name, period))
        return documents

    try:
        before_docs, after_docs = read(args.before, "before"), read(args.after, "after")
        result = analyze_documents(before_docs, after_docs)
        run_comparison_agent(
            before_docs, after_docs, result, settings.api_key,
            model=args.model or settings.model,
            progress=lambda index, total, message: print(f"{index}/{total}: {message}", flush=True),
        )
        review = result.ai_review
        print(json.dumps({key: review.get(key) for key in (
            "status", "model", "completed_batches", "total_batches", "covered_sources",
            "total_sources", "input_tokens", "output_tokens", "rejected_items", "error",
        )}, ensure_ascii=False))
        print(f"Validated comparison highlights: {len(review.get('comparisons', []))}")
        if args.output:
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
            print(f"Report saved: {args.output}")
        return 0 if review.get("status") in {"completed", "partial"} else 2
    except Exception as error:
        # Raw SDK/parser errors can include request data or credentials.
        print(f"Comparison could not finish ({type(error).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
