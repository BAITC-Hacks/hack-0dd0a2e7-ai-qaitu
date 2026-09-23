import unittest

from qaitu.extractors import _pdf_fragments, _pdf_page_text


class FakePage:
    def extract_text(self, extraction_mode="plain"):
        if extraction_mode == "layout":
            return (
                "3.4. БВА состоит из следующих структурных подразделений:\n"
                "а. Департамент непрерывного мониторинга системы внутреннего контроля (ДНМ).\n"
                "б. Департамент контроля качества аудита и методологии (ДККМ).\n"
                "3.5. Главному аудитору подчиняются работники БВА.\n"
            )
        return "\n".join(["3.4.", "БВА", "состоит"] + ["слово"] * 25)


class ExtractorTest(unittest.TestCase):
    def test_word_per_line_pdf_uses_layout_and_separates_organization_items(self):
        fragments = [part.strip() for part in _pdf_fragments(_pdf_page_text(FakePage())) if part.strip()]
        self.assertEqual(len(fragments), 4)
        self.assertTrue(fragments[1].startswith("а. Департамент"))
        self.assertTrue(fragments[2].startswith("б. Департамент"))


if __name__ == "__main__":
    unittest.main()
