import argparse
import glob
import json
import logging
import logging.config
import shutil
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TextSplitter

from scripts.bot.indexer.sklearn_indexer import SKLearnIndexer
from scripts.bot.loader.doc_csv_loader import DocCsvLoder


# import sklearn.metrics

def ingest_pdf_file(pdf_file: str, text_splitter: TextSplitter):
    logging.debug(f"Ingesting pdf file {pdf_file}")

    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split(text_splitter=text_splitter)

    return pages


def ingest_pdf_data(pdf_data_path: Path, text_splitter: TextSplitter):
    search_pattern = f"{pdf_data_path}/*.pdf"
    pdf_files = glob.glob(search_pattern)

    pdf_docs = []
    for pdf_file in pdf_files:
        pdf_data = ingest_pdf_file(pdf_file, text_splitter)
        pdf_docs.extend(pdf_data)

    return pdf_docs


def ingest_doc_csv_file(csv_path: Path, text_splitter: TextSplitter):
    logging.debug(f"Ingesting csv file {csv_path}")

    docs = DocCsvLoder(csv_path).load()
    docs = text_splitter.split_documents(docs)

    return docs


def ingest_doc_csv_data(csv_path: Path, text_splitter: TextSplitter):
    search_pattern = f"{csv_path}/*.csv"
    files = glob.glob(search_pattern)

    all_docs = []
    for file in files:
        docs = ingest_doc_csv_file(Path(file), text_splitter)
        all_docs.extend(docs)

    return all_docs


def ingest_data(data_path: Path, text_splitter: TextSplitter):
    # pdf_docs = ingest_pdf_data(data_path / "pdf", text_splitter)
    doc_csv = ingest_doc_csv_data(data_path / "doc_csv", text_splitter)
    return doc_csv


def cache_documents(docs: List[Document], cache_path: Path):
    if not cache_path.exists():
        cache_path.mkdir(parents=True)

    name = "document"
    docs_dict = [(doc.page_content, doc.metadata) for doc in docs]
    with open(cache_path / f'{name}.json', 'w') as f:
        json.dump(docs_dict, f, indent=4)


def index_data(docs: List[Document], embeddings: Embeddings, index_path: Path):
    if index_path.exists(): shutil.rmtree(index_path)

    SKLearnIndexer.Hybrid.save_index(docs, embeddings, index_path)


def main():
    logging.config.fileConfig('logging.ini')

    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to read data from')
    parser.add_argument('--cache_dir', type=str, default='run/cache', help='Cache directory')
    parser.add_argument('--index_dir', type=str, default='run/index', help='Directory to save index')
    parser.add_argument('--st_model_name', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='sentence-transformer embedding model')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Large text will be split into chunks of this size. 0 means model maximum allowed.')
    parser.add_argument('--chunk-overlap', type=int, default=20, help='Chunks will have overlap of this size.')

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir)
    index_path = Path(args.index_dir)
    st_model_name = args.st_model_name
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    logging.debug(f"Reading data from directory: {data_path}")
    logging.debug(f"Cache directory: {cache_path}")
    logging.debug(f"Index directory: {index_path}")
    logging.debug(f"Embedding model name: {st_model_name}")

    # 'max_seq_length': 128, 384 dimensional dense vector
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # most downloads in HF

    # 'max_seq_length': 256, 384 dimensional dense vector
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"  # most likes in HF

    # 'max_seq_length': 384, 768 dimensional dense vector
    # model_name = "sentence-transformers/all-mpnet-base-v2"  # default

    text_splitter = SentenceTransformersTokenTextSplitter(model_name=st_model_name, chunk_overlap=chunk_overlap,
                                                          tokens_per_chunk=chunk_size)
    logging.debug(f"Text chunk length: {text_splitter.tokens_per_chunk}")

    docs = ingest_data(data_path, text_splitter)

    if docs is None or len(docs) == 0:
        print("No data to index")
        return

    cache_documents(docs, cache_path)

    dense_embedder = HuggingFaceEmbeddings(model_name=st_model_name)
    index_data(docs, dense_embedder, index_path)


if __name__ == "__main__":
    main()
