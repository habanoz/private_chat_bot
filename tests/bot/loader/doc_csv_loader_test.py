import unittest
from pathlib import Path

from scripts.bot.loader.doc_csv_loader import DocCsvLoder


class DocCsvLoderTest(unittest.TestCase):
    def test_load(self):
        data = DocCsvLoder(Path("notebooks/squad-train-docs.csv")).load()
        self.assertIsNotNone(data)


if __name__ == '__main__':
    unittest.main()