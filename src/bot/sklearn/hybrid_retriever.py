import scipy
import pickle
from pathlib import Path
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.retrievers import TFIDFRetriever
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import SKLearnVectorStore


class SKLearnHybridRetriever(BaseRetriever):

    def __init__(self, vector_store: SKLearnVectorStore,
                 sparse_retriever: TFIDFRetriever,
                 embeddings: Embeddings,
                 k: int = 4):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.embeddings = embeddings
        self.k = k

    @classmethod
    def from_documents(cls, docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
        tfid_retriever = TFIDFRetriever.from_documents(docs, k=k)

        vector_store = SKLearnVectorStore(embeddings, persist_path=str(index_path / "dense" / "data.json"))
        vector_store.add_documents(docs)

        return cls(vector_store=vector_store, sparse_retriever=tfid_retriever, embeddings=embeddings, k=k)

    @classmethod
    def from_index(cls, docs: List[Document], embeddings: Embeddings, index_path: Path, k=4):
        with open(str(index_path / "sparse" / 'vectorizer.pkl'), 'rb') as file:
            vectorizer = pickle.load(file)

        with open(str(index_path / "sparse" / 'tfidf_array.npz'), 'rb') as file:
            tfidf_array = scipy.sparse.load_npz(file)

        tfid_retriever = TFIDFRetriever(vectorizer=vectorizer, tfidf_array=tfidf_array, docs=docs, k=k)
        vector_store = SKLearnVectorStore(embeddings, persist_path=str(index_path / "dense" / "data.json"))

        return cls(vector_store=vector_store, sparse_retriever=tfid_retriever, embeddings=embeddings, k=k)

    def save(self, index_path: Path):
        if not index_path.exists():
            index_path.mkdir()
            (index_path / "sparse").mkdir()
            (index_path / "dense").mkdir()

        self.vector_store.persist()

        with open(index_path / "sparse" / 'vectorizer.pkl', 'wb') as file:
            pickle.dump(self.sparse_retriever.vectorizer, file)

        with open(index_path / "sparse" / 'tfidf_array.npz', 'wb') as file:
            scipy.sparse.save_npz(file, self.sparse_retriever.tfidf_array)

    def get_relevant_documents(self, query: str) -> List[Document]:
        vector_store_retriever = self.vector_store.as_retriever(search_kwargs={'k': self.k})
        vector_fetched = vector_store_retriever.get_relevant_documents(query)

        keyword_fetched = self.sparse_retriever.get_relevant_documents(query)

        unique_docs_dict = {doc.page_content: doc for doc in vector_fetched + keyword_fetched}

        return list(unique_docs_dict.values())

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
