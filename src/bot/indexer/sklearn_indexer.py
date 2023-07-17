from pathlib import Path
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import SKLearnVectorStore

from src.bot.sklearn.hybrid_retriever import SKLearnHybridRetriever


class SKLearnIndexer:
    class Hybrid:
        @staticmethod
        def save_index(docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
            retriever = SKLearnHybridRetriever.from_documents(docs, embeddings, index_path, k=k)
            retriever.save(index_path)

    class Dense:
        @staticmethod
        def save_index(docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
            vector_store = SKLearnVectorStore(embeddings, persist_path=str(index_path / "dense" / "data.json"))
            vector_store.add_documents(docs)
            vector_store.persist()
