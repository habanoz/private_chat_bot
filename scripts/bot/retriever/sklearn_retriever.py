from pathlib import Path
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import SKLearnVectorStore

from scripts.bot.sklearn.hybrid_retriever import SKLearnHybridRetriever


class SKLearnRetriever:
    class Hybrid:
        @staticmethod
        def from_index(docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
            return SKLearnHybridRetriever.from_index(docs, embeddings, index_path, k=k)

    class Dense:
        @staticmethod
        def from_index(docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
            vector_store = SKLearnVectorStore(embeddings, persist_path=str(index_path / "dense" / "data.json"))
            return vector_store.as_retriever(search_kwargs={'k': k})
