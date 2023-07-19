import os
from pathlib import Path

import langchain
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from scripts.bot.loader.doc_csv_loader import DocCsvLoder
from scripts.bot.retriever.sklearn_retriever import SKLearnRetriever

langchain.debug=True
os.environ['DEBUG'] = 'True'

MODEL_NAMES = ["togethercomputer/RedPajama-INCITE-Chat-3B-v1", "tiiuae/falcon-7b-instruct"]
MODEL_NAME = MODEL_NAMES[1]
def main():
    docs = DocCsvLoder(Path("data/doc_csv/squad-validation-docs-unique.csv")).load()

    dense_embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = SKLearnRetriever.Dense.from_index(docs, dense_embedder, Path("run/index"), k=1)

    question = "What areas did Beyonce compete in when she was growing up?"

    # Encode the stop token using the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    stop_tokens =[ 'Question:', '<human>:','<bot>:']
    stop_token_ids = [tokenizer.encode(stop_token, add_special_tokens=False)[0] for stop_token in stop_tokens]

    llm = HuggingFacePipeline.from_model_id(model_id=MODEL_NAME,
                                            task="text-generation",
                                            model_kwargs={"temperature": 0,
                                                          # "max_length": 1024,
                                                          # "eos_token_id": stop_token_ids
                                                          }
                                            )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)

    print(answer)


if __name__ == '__main__':
    main()
