#!/usr/bin/env python3

import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline


def process_abstract(abstract, prompt_template, pipe, generation_args):
    uri = abstract["uri"]
    label = abstract["label"]
    abstract = abstract["abstract"]
    prompt = prompt_template.format(uri=uri, label=label, abstract=abstract)
    message = [{"role": "user", "content": prompt}]
    output = pipe(message, **generation_args)[0]["generated_text"]
    return output

def append_json(dict_, output):
    with open(output, "a") as f:
        f.write(json.dumps(dict_)+"\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        default="./data/main_graph_abstracts.jsonl",
                        help="file containing the DBpedia abstracts from which to extract triples")
    parser.add_argument("-p", "--prompt", type=str,
                        default="./prompts/prompt_abstract.txt",
                        help="file with the prompt to pass to an LLM")
    parser.add_argument("-o", "--output", type=str,
                        default="./data/main_graph_abstracts_triples.jsonl",
                        help="filename where the output will be saved")
    parser.add_argument("-m", "--model", type=str,
                        default="google/gemma-2-9b-it",
                        help="model used to extract triples")
    args = parser.parse_args()

    print("Loading prompt")
    with open(args.prompt, "r") as p:
        prompt_template = p.read().strip()

    print("Loading already processed URIs")
    processed_uris = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                d = json.loads(line.strip())
                processed_uris.add(d["uri"])

    print("Loading abstracts")
    abstracts_file = args.input
    abstracts = []
    with open(abstracts_file, "r") as f:
        for line in f:
            abstracts.append(json.loads(line.strip()))

    # Load model
    print("Loading model")
    model_name = args.model
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda",
                                                 torch_dtype="auto",
                                                 config=model_config,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generation_args = {"return_full_text": False, "max_new_tokens": 512}


    pbar = tqdm(abstracts)
    for abstract in pbar:
        uri = abstract["uri"]
        if uri in processed_uris:
            pbar.set_description(f"Skipping {uri}")
            continue
        pbar.set_description(f"Processing {uri}")
        output = process_abstract(abstract, prompt_template, pipe, generation_args)
        append_json({"uri": uri, "llm_output": output}, args.output)
