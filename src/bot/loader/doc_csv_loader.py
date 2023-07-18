from pathlib import Path
from langchain.schema import Document
import json
import pandas as pd


class DocCsvLoder:
    def __init__(self, path: Path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        result = [Document(page_content=row.page_content, metadata=json.loads(row.metadata)) for row in df.itertuples(index=False)]

        return result
