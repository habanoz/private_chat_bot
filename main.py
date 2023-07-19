import os
from pathlib import Path

import langchain
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from scripts.bot.loader.doc_csv_loader import DocCsvLoder
from scripts.bot.retriever.sklearn_retriever import SKLearnRetriever

langchain.debug = True
os.environ['DEBUG'] = 'True'

MODEL_NAMES = ["togethercomputer/RedPajama-INCITE-Chat-3B-v1", "tiiuae/falcon-7b-instruct"]
MODEL_NAME = MODEL_NAMES[1]


def main():
    docs = DocCsvLoder(Path("data/doc_csv/squad-validation-docs-unique.csv")).load()

    dense_embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = SKLearnRetriever.Dense.from_index(docs, dense_embedder, Path("run/index"), k=1)

    question = "What areas did Beyonce compete in when she was growing up?"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, load_in_8bit=True, device_map="auto"
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    generation_config = model.generation_config
    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    stop_tokens = ['Question:', '<human>:', '<bot>:']
    stop_token_ids = [tokenizer.encode(stop_token, add_special_tokens=False)[0] for stop_token in stop_tokens]

    generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        # stopping_criteria=stopping_criteria,
        generation_config=generation_config,
        odel_kwargs={"temperature": 0,
                     # "max_length": 1024,
                     # "eos_token_id": stop_token_ids
                     }
    )

    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)

    print(answer)


if __name__ == '__main__':
    main()
