from langchain.schema import Document
from pathlib import Path
from typing import List
import logging
import json


class DocumentCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path

    def save(self, docs: List[Document]):
        if not self.cache_path.exists(): self.cache_path.mkdir(parents=True)

        docs_dict = [(doc.page_content, doc.metadata) for doc in docs]
        with open(self.cache_path / 'document.json', 'w') as f:
            json.dump(docs_dict, f, indent=4)

    def load(self) -> List[Document]:
        if not self.cache_path.exists():
            logging.debug(f"Cache dir {self.cache_path} does not exist!")
            return []

        with open(self.cache_path / 'document.json', 'r') as f:
            json_docs = json.load(f)

        docs_dict = [Document(page_content=doc[0], metadata=doc[1]) for doc in json_docs]
        return docs_dict
