from langchain.document_loaders import PyPDFLoader
from pathlib import Path
import logging
import glob

from scripts.bot.loader.doc_csv_loader import DocCsvLoder


class DataLoader:

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load(self):
        pdf_docs = self._load_pdf_data(self.data_path / "pdf")
        doc_csv = self._load_doc_csv_data(self.data_path / "doc_csv")

        return pdf_docs + doc_csv

    def _load_pdf_data(self, pdf_data_path: Path):
        search_pattern = f"{pdf_data_path}/*.pdf"
        pdf_files = glob.glob(search_pattern)

        pdf_docs = []
        for pdf_file in pdf_files:
            logging.debug(f"Ingesting pdf file {pdf_file}")

            pdf_doc = PyPDFLoader(pdf_file).load()
            pdf_docs.append(pdf_doc)

        return pdf_docs

    def _load_doc_csv_data(self, csv_path: Path):
        search_pattern = f"{csv_path}/*.csv"
        files = glob.glob(search_pattern)

        all_docs = []
        for file in files:
            logging.debug(f"Ingesting doc csv file {file}")

            docs = DocCsvLoder(Path(file)).load()
            all_docs.extend(docs)

        return all_docs
