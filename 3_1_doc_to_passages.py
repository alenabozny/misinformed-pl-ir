#!/usr/bin/env python

import json
import nltk
import os
import pandas as pd
import re
nltk.download('punkt_tab')

from multiprocessing import Pool
from tqdm import tqdm


def contains_punctuation(word):
    return bool(re.search(r"[!,.:;?]", word))

def remove_useless_lines(doc):
    new_doc = []
    for line in doc.split("\n"):
        if contains_punctuation(line):
            no_section_number = re.sub(r"^\d+(\.\d+)*\s+", '', line)
            if contains_punctuation(no_section_number):
                new_doc.append(line)
    return "\n".join(new_doc)

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def subsencentes(sentence, new_sentence_len=200):
    chunks = divide_chunks(sentence.split(), new_sentence_len)
    return [" ".join(chunk) for chunk in chunks]

def split_into_passages(docidtext, tokens_per_passage: int = 300, n_overlap_sentences=1):
    docid, text = docidtext
    text = remove_useless_lines(text)
    text = re.sub(r'http[s]?://\S+', '', text)

    original_sentences = nltk.sent_tokenize(text)
    # split very long sentences in "subsentences"
    sentences = []
    for sentence in original_sentences:
        if len(sentence.split()) < 300:
               sentences.append(sentence)
        else:
               sentences.extend(subsencentes(sentence))

    passages = []
    current_passage = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = re.findall(r'\S+', sentence)
        sentence_token_count = len(sentence_tokens)

        if current_tokens + sentence_token_count > tokens_per_passage and current_passage:
            passage_text = " ".join(current_passage)
            passages.append(passage_text)

            # Start a new passage with overlap
            overlap_sentences = current_passage[-n_overlap_sentences:]  # Use last sentence for overlap
            current_passage = overlap_sentences
            current_tokens = sum(len(re.findall(r'\S+', s)) for s in overlap_sentences)

        current_passage.append(sentence)
        current_tokens += sentence_token_count

    # Add the last passage if there's anything left
    if current_passage:
        passage_text = " ".join(current_passage)
        passages.append(passage_text)

    passages = {f"passage_{i+1}": passage  for i, passage in enumerate(passages)}
    return {"doc_id": docid, "passages": passages}

def convert_to_passages(input_file, preprocess_docid, columns):
    print(f"Loading input file: {input_file}")
    docs = pd.read_csv(input_file, header=None)
    docs.columns = columns
    docs = docs[["docid", "text"]]

    print("Splitting into passages")
    with Pool(8) as pool:
        doc_passages = pool.map(split_into_passages, docs.values)

    base_out_file = os.path.basename(input_file)
    out_dir = "./data"
    output_file = os.path.join(out_dir, base_out_file.replace(".csv", "_passages.jsonl"))

    print("Saving results")
    with open(output_file, "w") as f:
        for doc in tqdm(sorted(doc_passages, key=lambda x: x["doc_id"])):
            doc["doc_id"] = preprocess_docid(doc["doc_id"])
            f.write(json.dumps(doc)+"\n")


if __name__ == "__main__":

    input_file = "./data/trec2020_first_stage_retrieved_docs.csv"
    preprocess_docid =  lambda x: x.strip("<|>").replace("urn:uuid:", "")
    columns = ["docid", "text"]
    convert_to_passages(input_file, preprocess_docid, columns)

    input_file = "./data/trec2021-22_first_stage_retrieved_docs.csv"
    preprocess_docid =  lambda x: x
    columns = ["docid", "text", "url", "timestamp"]
    convert_to_passages(input_file, preprocess_docid, columns)
