import argparse
import logging
import logging.config
import os
import csv
from pathlib import Path

import langchain
from langchain import HuggingFacePipeline, PromptTemplate, OpenAI, LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from bot.loader.doc_csv_loader import DocCsvLoder
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
    parser.add_argument('--output_dir', type=str, default='run/test', help='Cache directory')
    parser.add_argument('--hf_model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Huggingface model name')
    parser.add_argument('--oai_model_name', type=str, default='text-davinci-003', help='OpenAI model name')
    parser.add_argument('--model_platform', type=str, default='hf', choices=['hf', 'oai'], help='Model platform')
    parser.add_argument('--device', type=str, default='auto', choices=["auto", "cpu", "cuda"], help='device')
    parser.add_argument('--use_8bit', action='store_true', help='8bit quantization for chat model')
    parser.add_argument('--run_name', type=str, default="{model_name}",
                        help='Give a name to this run. Helpful for testing.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Model config: max tokens to generate')
    parser.add_argument('--max_question_size', type=int, default=50, help='maximum question size')

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir)
    output_path = Path(args.output_dir)
    hf_model_name = args.hf_model_name
    oai_model_name = args.oai_model_name
    model_platform = args.model_platform
    device = args.device
    use_8bit = args.use_8bit
    max_new_tokens = args.max_new_tokens
    max_question_size = args.max_question_size

    model_name_str = oai_model_name if model_platform == "oai" else hf_model_name
    model_name_str = model_name_str.replace('-','_').replace('/','_')
    run_name = args.run_name.replace("{model_name}", model_name_str)

    logging.debug(f"Reading data from directory: {data_path}")
    logging.debug(f"Cache directory: {cache_path}")

    if model_platform == "oai":
        logging.debug(f"LLM: {oai_model_name}")
    else:
        logging.debug(f"LLM : {hf_model_name}")

    llm = OpenAI(model_name=oai_model_name) if model_platform == "oai" else get_hf_model(hf_model_name, use_8bit, device, 2048,max_new_tokens)

    with open("prompts/chat-prompt.txt", "r") as f:
        prompt_template = f.read().strip()

    max_length = llm.pipeline.model.config.max_length
    max_chunk_size = max_length - max_new_tokens - max_question_size - len(prompt_template.split())
    logging.debug(f"Max chunk size for context: {max_chunk_size}")

    tokenizer = llm.pipeline.tokenizer

    separator = '\\.'
    no_escape_separator = '.'

    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=max_chunk_size, chunk_overlap=0, separator=separator
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(prompt=prompt, llm=llm)

    docs = DocCsvLoder(Path("data/doc_csv/squad-validation-docs-unique.csv")).load()
    docs = text_splitter.split_documents(docs)
    docs = [Document(page_content=doc.page_content.replace(separator, no_escape_separator), metadata=doc.metadata) for
            doc in docs]

    rows = []

    for doc in docs[:1]:
        context = doc.page_content
        for question, answer in zip(doc.metadata['question'], doc.metadata['answers']):
            model_answer = chain.run({"context": context, "question": question})
            rows.append({'context': context, 'question': question, 'answer': answer, 'model_answer': model_answer})

            print(model_answer)

    csv_path = output_path / run_name / "out.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['context', 'question', 'answer', 'model_answer'])
        writer.writeheader()
        writer.writerows(rows)


def get_hf_model(model_name: str, use_8bit: bool, device: str, max_length:int, max_new_tokens: int):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        load_in_8bit=use_8bit,
        device_map=device,
        max_length=max_length
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, model_max_length=max_length)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
    model = model.eval()
    logging.debug(f"Model max sequence length {model.config.max_length}")

    generation_config = model.generation_config
    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = model.config.max_length
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # stop_tokens = ['Question:', '<human>:', '<bot>:']
    # stop_token_ids = [tokenizer.encode(stop_token, add_special_tokens=False)[0] for stop_token in stop_tokens]

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
