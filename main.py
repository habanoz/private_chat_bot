import os
from pathlib import Path

import langchain
from langchain import HuggingFacePipeline, PromptTemplate, OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from scripts.bot.loader.doc_csv_loader import DocCsvLoder
from scripts.bot.retriever.sklearn_retriever import SKLearnRetriever
import os

langchain.debug = True
os.environ['DEBUG'] = 'True'
os.environ['OPENAI_API_KEY'] = ""

MODEL_NAMES = ["togethercomputer/RedPajama-INCITE-Chat-3B-v1", "tiiuae/falcon-7b-instruct"]
MODEL_NAME = MODEL_NAMES[0]


def main():
    docs = DocCsvLoder(Path("data/doc_csv/squad-validation-docs-unique.csv")).load()

    dense_embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    retriever = SKLearnRetriever.Dense.from_index(docs, dense_embedder, Path("run/index"), k=1)

    question = "What areas did Beyonce compete in when she was growing up?"

    llm = get_hf_model(MODEL_NAME)
    # llm = OpenAI()

    prompt_template = """Use the following pieces of context delimited by triple backticks to answer the question at the end. If the context does not contain the answer, do not try to generate answer, just say that the provided context does not include the answer.

    Context: ```{context}```

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs = {"prompt": prompt})
    answer = qa.run(question)

    print(answer)


def get_hf_model(model_name:str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, load_in_8bit=True, device_map="auto"
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        model_kwargs={"temperature": 0,
                      # "max_length": 1024,
                      # "eos_token_id": stop_token_ids
                      }
    )

    return HuggingFacePipeline(pipeline=generation_pipeline)


if __name__ == '__main__':
    main()
