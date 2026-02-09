#!/usr/bin/env python3

import json
import os

from rdflib import URIRef, Literal
from retriv import DenseRetriever

from hkg.graph_utils import load_named_graph
from hkg.labels import Labels, URI2labels

import logging
logging.getLogger("rdflib").setLevel(logging.ERROR)

graph_name = "full_graph"
print(f"Loading the graph '{graph_name}'...")
graph = load_named_graph(graph_name)

index_name = f"{graph_name}_labels"
collection_filename = f"./data/labels/{graph_name}_labels.jsonl"
os.makedirs("./data/labels/", exist_ok=True)
print(f"Computing {graph_name} entities labels...")
labels = Labels(graph)

print(f"Collecting graph entities labels to index...")
terms_to_index = set()
for term in labels.hkg_entities:
    if isinstance(term, URIRef):
        terms_to_index.update(URI2labels(term))
    if isinstance(term, Literal):
        if term.value and not isinstance(term.value, int):
            terms_to_index.add(term.value)

print(f"Saving collection of entities labels to {collection_filename}...")
with open(collection_filename, "w") as f:
    for i, label in enumerate(sorted(terms_to_index)):
        d = {"id": f"doc_{i+1}", "text": label}
        f.write(json.dumps(d)+"\n")

print("Indexing graph entities...")
dr = DenseRetriever(index_name=index_name, model="abhinand/MedEmbed-small-v0.1",
                    normalize=True, max_length=128, use_ann=False)
df = dr.index_file(path=collection_filename, embeddings_path=None,
                   use_gpu=False, batch_size=512, show_progress=True)
print("Done.")
