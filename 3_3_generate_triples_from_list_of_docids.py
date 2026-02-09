#!/usr/bin/env python3

import argparse
import json
import os

from datetime import datetime

from hkg.rdf_regex import parse_triples_iter


now = datetime.now

def load_prompt_template(prompt_file):
    with open(prompt_file, "r") as p:
        prompt_template = p.read().strip()
    return prompt_template

def load_top_ids(file, depth):
    output = []
    with open(file, "r") as f:
        for line in f:
            rank, docid = line.strip().split(",")
            if int(rank) > depth:
                break
            output.append(docid)
    return output

def load_relevant_passages(doc_ids, passages_file):
    passages = {}
    with open(passages_file, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            if d["doc_id"] in doc_ids:
                passages[d["doc_id"]] = d["passages"]
    return passages

def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda",
                                               torch_dtype="auto", config=model_config,
                                               trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"Loaded model {model_name}.")
    return pipe

def generate_response(llm_input, pipe=None, test_mode=False):
    if test_mode:
        # Don't extract triples, just return the prompt
        return llm_input
    messages = [{"role": "user", "content": llm_input}]
    generation_args = {"max_new_tokens": 512, "return_full_text": False,
                       "truncation": True}
    generated_text = pipe(messages, **generation_args)[0]["generated_text"]
    return generated_text

# def touch(file):
#     with open(file, "w") as f:
#         f.write("")

def touch(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")

def output_is_unrelated(text):
    matches = set(parse_triples_iter(text.strip()))
    if not matches and "UNRELATED" in text:
        return True
    return False

def main(args):
    input_file = args.input
    prompt_template = load_prompt_template(args.prompt)

    # Load IDs of documents from which to extract the RDF triples
    doc_ids = load_top_ids(input_file, args.depth)

    # Load the passages of the relevant documents
    passages = load_relevant_passages(doc_ids, args.passages)
    n_docs = len(doc_ids)

    # In test mode, this variable will not be used.
    pipe = None
    if not args.test:
        pipe = load_model(args.model)

    for doc_counter, doc in enumerate(doc_ids):
        output_file = os.path.join(args.outdir, doc)

        if args.test:
            # Add another extension for the files created in test mode to avoid confusion
            output_file = output_file+".TEST"

        # If the output file already exists, skip the document
        if os.path.exists(output_file):
            print(f"Skipping {doc}: output already saved at {output_file} or currently being processed by another script")
            continue

        # Otherwise, extract triples from the document
        print(f"{now()} Working on doc {doc}: {doc_counter+1}/{n_docs}")

        # Touch the output file, so that other instances of this script running in
        # parallel won't attempt to work on the same doc
        touch(output_file)

        # Initialize output dict
        output_dict = {"doc_id": doc, "passages" : {} }
        n_passages = len(passages[doc])
        n_unrelated = 0

        for passage_counter, (passage_id, passage) in enumerate(passages[doc].items()):

            if len(passage) > 15000:
                # Some passages might be too long, and usually only contain garbage.
                llm_output = "SKIPPED"
            else:
                print(f"\t{now()} Working on {passage_id}: {passage_counter+1}/{n_passages}")
                if (n_unrelated/n_passages) > 0.5:
                    # If most of the passages of the doc processed so far are UNRELATED,
                    # the other passages are marked as UNRELATED by default to save
                    # resources
                    print(f"Outputting UNRELATED by default as {n_unrelated}/{n_passages}>0.5")
                    llm_output = "UNRELATED"
                else:
                    # Otherwise, the LLM is used to extract triples from the passage
                    llm_input = prompt_template.format(passage=passage)
                    llm_output = generate_response(llm_input, pipe, args.test)
                    if output_is_unrelated(llm_output):
                        # Keep count of the passages marked as UNRELATED by the LLM, i.e.,
                        # passages that do not contain medical information to extract
                        # triples from
                        print(f"\t{now()} Output is UNRELATED: {llm_output}")
                        n_unrelated += 1
            key_name = "rdf_"+passage_id
            output_dict["passages"][key_name] = llm_output

        print(f"{now()} Saving the extracted triples to {output_file}")
        with open(output_file, "w") as f:
            f.write(json.dumps(output_dict)+"\n")


if __name__ == "__main__":

    from huggingface_hub import login

    login(token="your-hf-token")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        #default="./data/top_docs/2020/run.misinfo-2020-description.bm25_en/query_1_top_docs.txt",
                        default="./data/top_docs/custom/custom_bm25_results/query_215_top_docs.txt",
                        help="file with a list of ranked document IDs from which to extract triples")
    parser.add_argument("-d", "--depth", type=int, default=200,
                        help="only the top 'depth' documents in the file will be processed")
    parser.add_argument("--passages", type=str,
                        #default="./data/trec2020_first_stage_retrieved_docs_passages.jsonl",
                        default="./data/custom_docs_passages.jsonl",
                        help="file containing the document passages")
    parser.add_argument("--outdir", type=str, default="./triples/custom",
                        help="folder in which to save the files with extracted triples for each document")
    parser.add_argument("-p", "--prompt", type=str,
                        default="./prompts/prompt_document.txt",
                        help="file with the prompt to feed to the LLM")
    parser.add_argument("-m", "--model", type=str, default="google/gemma-2-9b-it",
                        help="name of the model to use to extract triples")
    parser.add_argument("-t", "--test", action="store_true",
                        help="test mode: test the script without extracting triples")
    args = parser.parse_args()

    main(args)
