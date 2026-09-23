import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qaitu.ai_reviewer import review_with_llm
from qaitu.analyzer import analyze_documents
from qaitu.demo import demo_documents


class AIReviewerTest(unittest.TestCase):
    def test_accepts_real_sources_and_discards_invented_source(self):
        before, after = demo_documents()
        base = analyze_documents(before, after)
        source_id = before[0].fragments[0].id
        response = SimpleNamespace(
            status="completed",
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")])],
            output_text=json.dumps({"findings": [
                {"kind": "loss", "title": "Проверить функцию", "source_ids": [source_id], "confidence": 0.7},
                {"kind": "loss", "title": "Выдуманный источник", "source_ids": ["not-in-catalog"]},
            ]}),
        )
        with patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value = response
            findings = review_with_llm(before, after, base, "test-key")
            self.assertFalse(client.return_value.responses.create.call_args.kwargs["store"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].sources[0].id, source_id)


if __name__ == "__main__":
    unittest.main()
