import unittest

from qaitu.analyzer import analyze_documents
from qaitu.demo import demo_documents


class AnalyzerTest(unittest.TestCase):
    def test_demo_detects_required_cases(self):
        before, after = demo_documents()
        result = analyze_documents(before, after)
        kinds = {finding.kind for finding in result.findings}
        self.assertTrue({"loss", "duplicate", "conflict"} <= kinds)
        self.assertTrue(all(finding.sources for finding in result.findings))
        self.assertTrue(any(change.status in {"created", "transformed"} for change in result.unit_changes))

    def test_every_source_is_traceable(self):
        before, after = demo_documents()
        result = analyze_documents(before, after)
        for finding in result.findings:
            for source in finding.sources:
                self.assertTrue(source.id)
                self.assertTrue(source.document)
                self.assertTrue(source.locator)
                self.assertTrue(source.text)


if __name__ == "__main__":
    unittest.main()
