import argparse
import logging
import logging.config
import os
from pathlib import Path

import langchain
from langchain import HuggingFacePipeline, PromptTemplate, OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from bot.loader.document_cache import DocumentCache
from bot.retriever.sklearn_retriever import SKLearnRetriever

langchain.debug = True
os.environ['DEBUG'] = 'True'
os.environ['OPENAI_API_KEY'] = ""


def main():
    logging.config.fileConfig('logging.ini')

    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to read data from')
    parser.add_argument('--cache_dir', type=str, default='run/cache', help='Cache directory')
    parser.add_argument('--index_dir', type=str, default='run/index', help='Directory to save index')
    parser.add_argument('--st_model_name', type=str, default='sentence-transformers/all-mpnet-base-v2', help='sentence-transformer embedding model')
    parser.add_argument('--hf_model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Huggingface model name')
    parser.add_argument('--oai_model_name', type=str, default='text-davinci-003', help='OpenAI model name')
    parser.add_argument('--model_platform', type=str, default='hf', choices=['hf', 'oai'], help='Model platform')
    parser.add_argument('--k', type=int, default=1, help='retrieval k')
    parser.add_argument('--question', type=str, default='What areas did Beyonce compete in when she was growing up?',
                        help='Question')
    parser.add_argument('--device', type=str, default='auto', choices=["auto", "cpu", "cuda"], help='device')
    parser.add_argument('--use_8bit', action='store_true', help='8bit quantization for chat model')
    parser.add_argument('--run_name', type=str, default="{model_name}", help='Give a name to this run. Helpful for testing.')

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir)
    index_path = Path(args.index_dir)
    st_model_name = args.st_model_name
    hf_model_name = args.hf_model_name
    oai_model_name = args.oai_model_name
    model_platform = args.model_platform
    k = args.k
    question = args.question
    device = args.device
    use_8bit = args.use_8bit

    logging.debug(f"Reading data from directory: {data_path}")
    logging.debug(f"Cache directory: {cache_path}")
    logging.debug(f"Index directory: {index_path}")
    logging.debug(f"Embedding model name: {st_model_name}")

    if model_platform == "oai":
        logging.debug(f"LLM: {oai_model_name}")
    else:
        logging.debug(f"LLM : {hf_model_name}")

    docs = DocumentCache(cache_path=cache_path).load()

    dense_embedder = HuggingFaceEmbeddings(model_name=st_model_name)
    retriever = SKLearnRetriever.Dense.from_index(docs, dense_embedder, index_path, k=k)

    llm = OpenAI(model_name=oai_model_name) if model_platform == "oai" else get_hf_model(hf_model_name, use_8bit, device)

    with open("prompts/chat-prompt.txt", "r") as f:
        prompt_template = f.read().strip()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs={"prompt": prompt})

    answer = qa.run(question)
    print(answer)


def get_hf_model(model_name: str, use_8bit: bool, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        load_in_8bit=use_8bit,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
    model = model.eval()
    logging.debug(f"Model max sequence length {model.config.max_length}")

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
