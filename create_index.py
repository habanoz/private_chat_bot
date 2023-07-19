import argparse
import json
import logging
import logging.config
import shutil
from pathlib import Path
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, TextSplitter

from bot.loader.data_loader import DataLoader
from bot.loader.document_cache import DocumentCache
from scripts.bot.indexer.sklearn_indexer import SKLearnIndexer


def ingest_data(data_path: Path, text_splitter: TextSplitter):
    docs = DataLoader(data_path).load()
    docs = text_splitter.split_documents(docs)
    return docs


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

    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=st_model_name, chunk_overlap=chunk_overlap, tokens_per_chunk=chunk_size)
    logging.debug(f"Text chunk length: {text_splitter.tokens_per_chunk}")

    docs = ingest_data(data_path, text_splitter)
    if docs is None or len(docs) == 0:
        print("No data to index!")
        return

    DocumentCache(cache_path=cache_path).save(docs)

    dense_embedder = HuggingFaceEmbeddings(model_name=st_model_name)
    index_data(docs, dense_embedder, index_path)


if __name__ == "__main__":
    main()
