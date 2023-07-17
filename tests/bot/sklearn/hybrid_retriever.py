import unittest
import shutil
import json
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.bot.sklearn.hybrid_retriever import SKLearnHybridRetriever


class SKLearnHybridRetrieverTest(unittest.TestCase):
    def test_create_index(self):
        with open("test/docs/docs.json") as f:
            docs = json.load(f)
            docs = [Document(page_content=doc[0], metadata=doc[1]) for doc in docs]

        embeddings = HuggingFaceEmbeddings()
        index_path = Path("test/index")

        shutil.rmtree(index_path)
        self.assertFalse((index_path / "dense").exists())

        retriever = SKLearnHybridRetriever.from_documents(docs, embeddings, index_path)
        retriever.save(index_path)

        self.assertTrue((index_path / "dense").exists())
        self.assertTrue((index_path / "dense/data.json").exists())

        self.assertTrue((index_path / "sparse").exists())
        self.assertTrue((index_path / "sparse/tfidf_array.npz").exists())
        self.assertTrue((index_path / "sparse/vectorizer.pkl").exists())

        retriever2 = SKLearnHybridRetriever.from_index(docs, embeddings, index_path, k=1)

        retrieved = retriever2.get_relevant_documents("What deos investment markets resemble?",)

        for doc in retrieved:
            print(doc.page_content)

        self.assertTrue(len(retrieved) in [2])


if __name__ == '__main__':
    unittest.main()
