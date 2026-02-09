#!/usr/bin/env python3

import json
import logging
import os

from tqdm import tqdm

from hkg.graph_utils import load_named_graph
from hkg.rdf_regex import parse_triples_iter
from hkg.labels import Labels


logging.getLogger("rdflib").setLevel(logging.ERROR)


def load_triples(triples_file):
    output = []
    with open(triples_file, "r") as f:
        for line in f:
            output.append(json.loads(line.strip()))
    return output

def triples_are_consistent(triples_regex_matches, dense=False):
    "Check whether the triples extracted from a document are consistent among each other"
    # TODO: consistency check does not account for inverse or similar predicates
    text_triples = {"TRUE": set(), "FALSE": set(), "ASK": set()}
    if not triples_regex_matches:
        return True
    for regex_match in triples_regex_matches:
        trip = regex_match.group()
        predicted, matches = labels._match_triple(trip, binary=False, dense=dense, cutoff=DENSE_CUTOFF)
        if not (predicted == "INVALID" or predicted == "UNKNOWN"):
            text_triples[predicted].update(matches)
    # If the intersection is empty: return True, else False.
    return not bool(text_triples["TRUE"].intersection(text_triples["FALSE"]))

def process_triple_matches(regex_matches, skip_duplicates=False, dense=False):
    """
    Verify the correctness of a triple against HKG. The following scores are computed:

     - `binary_correctness`: True or False
     - `binary_label`: TP, TN, FP or FN
     - `correctness`: True, False, INVALID or UNKNOWN
     - `binary_label`: TP, TN, FP, FN, INVALID or UNKNOWN

     The difference between `binary_correctness` and `correctness` is in the choice of the
     closed or open world assumption. If a triple is falsified by HKG simply because a
     term in the triple cannot be found in HKG, then `binary_correctness` will be False
     (closed world assmption, CWA),
     while `correctness` will be UNKNOWN (open world assumption, OWA). Similarly for
     `binary_label` and `label`.
    """
    triples_results = []
    # TODO: currently "TRUE" and "FALSE" are not updated or accessed
    global_results = { "TRUE": 0, "FALSE": 0, "CONSISTENT": triples_are_consistent(regex_matches, dense=dense) }

    # Keep a set of already processed triples to avoid scoring the same triple twice
    if skip_duplicates:
        already_processed_triples = set()
    for id_, regex_match in enumerate(regex_matches):
        trip = regex_match.group()
        if skip_duplicates:
            if trip in already_processed_triples:
                continue
            already_processed_triples.add(trip)
        predicted = regex_match.group("truth")
        subject = regex_match.group("subject")
        predicate = regex_match.group("predicate")
        object_ = regex_match.group("object")
        triple_results = {"triple": trip,
                          "subject": subject, "predicate": predicate, "object": object_}
        if dense:
            # Verify a triple also using semantic similarity matching
            output = labels.verify_triple_two_steps(trip, binary=False,
                                          allow_inverse=ALLOW_INVERSES,
                                          allow_similar_predicates=ALLOW_SIMILAR_PREDICATES, cutoff=DENSE_CUTOFF)
            binary_output = labels.verify_triple_two_steps(trip, binary=True,
                                          allow_inverse=ALLOW_INVERSES,
                                          allow_similar_predicates=ALLOW_SIMILAR_PREDICATES, cutoff=DENSE_CUTOFF)
        else:
            # Verify a triple matching its terms only through exact matching or synonyms
            output = labels.verify_triple(trip, binary=False,
                                          allow_inverse=ALLOW_INVERSES,
                                          allow_similar_predicates=ALLOW_SIMILAR_PREDICATES)
            binary_output = labels.verify_triple(trip, binary=True,
                                          allow_inverse=ALLOW_INVERSES,
                                          allow_similar_predicates=ALLOW_SIMILAR_PREDICATES)

        triple_results["binary_correctness"] = binary_output # always True or False (CWA)

        # 'output' might be 'INVALID' or 'UNKNOWN' (e.g., if a term cannot be matched in the HKG)
        # 'binary_output' is always True or False (closed world assumption)

        if output == "INVALID" or output == "UNKNOWN":
            # In this case, 'correctness' and 'label` are INVALID/UNKNOWN (OWA)
            triple_results["correctness"] = output
            triple_results["label"] = output
            # Only the 'binary_label` needs to be computed: TP, TN, FP or FN (CWA)
            if predicted == "TRUE": # the triple is TRUE (s p o)
                if binary_output: # TRUE (s p o) is verified: TP
                    triple_results["binary_label"] = "TP"
                else: # TRUE (s p o) is falsified: FP
                    triple_results["binary_label"] = "FP"
            elif predicted == "FALSE": # the triple is FALSE (s p o)
                if binary_output: # FALSE (s p o) is verified:
                    triple_results["binary_label"] = "TN"
                else:
                    triple_results["binary_label"] = "FN"
        else:
            assert binary_output == output
            triple_results["correctness"] = output
            if predicted == "TRUE": # the triple is TRUE (s p o)
                if output: # TRUE (s p o) is verified: TP
                    triple_results["label"] = "TP"
                else: # TRUE (s p o) is falsified: FP
                    triple_results["label"] = "FP"
            elif predicted == "FALSE": # the triple is FALSE (s p o)
                if output: # FALSE (s p o) is verified:
                    triple_results["label"] = "TN"
                else:
                    triple_results["label"] = "FN"
            else:
                raise Exception
            # Since the output is True or False, and binary_output == output,
            # the "binary_label" is the same as "label"
            triple_results["binary_label"] = triple_results["label"]
        triples_results.append(triple_results)
    return global_results, triples_results

def verify_doc_triples(doc, skip_duplicates=False, dense=False):
    triples = []
    unrelated = 0
    invalid = 0
    passages = doc["passages"]
    consistent = True
    n_passages=len(passages)
    for passage_id, passage in passages.items():
        matches = list(parse_triples_iter(passage))
        if not(matches):
            if "UNRELATED" in passage:
                unrelated += 1
            else:
                invalid += 1
        else:
            triples.extend(matches)
    if triples:
        global_results, triples = process_triple_matches(triples, skip_duplicates=skip_duplicates, dense=dense)
        consistent = global_results["CONSISTENT"]
    output = {
        "doc_id": doc["doc_id"],
        "n_unrelated_passages": unrelated, "n_invalid_passages": invalid,
        "n_passages": n_passages, "consistent": consistent,
        "triples": triples
    }
    return output

def get_output_file(base_file, skip_duplicates=False, dense=False, dense_cutoff=None):
    out_file = base_file
    if skip_duplicates:
        out_file =  out_file.replace(".json", "_skip-duplicates.json")
    if dense:
        out_file =  out_file.replace(".json", f"_dense-{dense_cutoff}-match.json")
    return out_file


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-g", "--graph", type=str, default="full_graph_abstract",
                        help="version of the HKG graph to load ('main_graph', 'full_graph' or 'full_graph_abstract'")
    parser.add_argument("-t", "--triples", type=str,
                        # default="./data/trec2020_first_stage_retrieved_docs_passages_RDF_triples.jsonl",
                        default="./triples/custom_concat.jsonl",
                        help="path of the file with the RDF triples to verify")
    parser.add_argument("-o", "--output", type=str, default="./data/correctness_scores/",
                        help="folder where to save the correctness scores")
    parser.add_argument("-d", "--dense", action="store_true",
                        help="whether to load the dense index to perform semantic similarity-based matching")
    parser.add_argument("-s", "--skip", action="store_true",
                        help="whether to skip duplicate triples")
    parser.add_argument("-c", "--cutoff", type=int, default=10,
                        help="if the dense index is used, how many similar terms to consider when matching")
    args = parser.parse_args()

    ALLOW_INVERSES = True
    ALLOW_SIMILAR_PREDICATES = True
    DENSE_CUTOFF = args.cutoff

    graph_name = args.graph
    print(f"Loading graph {graph_name}")
    graph = load_named_graph(graph_name)
    index_name = f"{graph_name}_labels"
    if args.dense:
        labels = Labels(graph, dense_index=index_name)
        labels.dense_retriever.encoder.change_device("cuda")
    else:
        labels = Labels(graph)

    triples_file = args.triples
    dense = args.dense
    skip_duplicates = args.skip
    docs = load_triples(triples_file)

    # Create the name of the output file based on the chosen options
    output_basename = os.path.basename(triples_file).replace(".jsonl", f"_{graph_name}_triple-level_correctness_scores.json")
    base_output_file = os.path.join(args.output, output_basename)
    output_file = get_output_file(base_output_file, skip_duplicates=skip_duplicates, dense=dense, dense_cutoff=DENSE_CUTOFF)

    sname = "skip-duplicates" if skip_duplicates else "no-skip-duplicates"
    dname = "dense" if dense else "no-dense"
    print(f"\nWorkin on {triples_file}: {sname}+{dname}")
    print(f"Output will be saved at {output_file}")

    results = []
    # Load document triples that have already been processed
    if os.path.exists(output_file):
        print("Loading already processed docs")
        with open(output_file, "r") as f:
            results = json.load(f)

    already_processed_docs = set()
    for doc in results:
        already_processed_docs.add(doc["doc_id"])

    for doc_counter, doc in tqdm(list(enumerate(docs))):
        if doc["doc_id"] in already_processed_docs:
            print(f"Skipping {doc['doc_id']}: already_processed")
            continue
        results.append(verify_doc_triples(doc, skip_duplicates=skip_duplicates, dense=dense))
        doc_counter += 1
        if doc_counter % 100 == 0:
            print(f"Saving partial output at {output_file}")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=1)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=1)
