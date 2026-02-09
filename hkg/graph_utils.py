#!/usr/bin/env python3

import pandas as pd
import pickle
import os
import re

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.util import from_n3

from hkg.namespaces import DBR, DBO, DBP, HKG
from hkg.rdf_regex import split_uri, string_is_URI


def add_namespaces(graph):
    graph.bind("dbr", DBR)
    graph.bind("dbo", DBO)
    graph.bind("dbp", DBP)
    graph.bind("hkg", HKG)

G = Graph()
add_namespaces(G)
NAMESPACE_MANAGER = G.namespace_manager

def subgraph(graph, predicates):
    "Return a subgraph containing only the specified predicates."
    if isinstance(predicates, str) or isinstance(predicates, URIRef):
        if "|" in predicates:
            predicates = predicates.split("|")
        else:
            predicates = [predicates]
    out_graph = Graph()
    add_namespaces(out_graph)
    for predicate in predicates:
        q = f"construct where {{ ?s {predicate} ?o }}"
        out_graph += graph.query(q).graph
    return out_graph

def save_graph(graph, graph_name, folder,
               save_pickle=True, save_turtle=False, save_csv=False):
    add_namespaces(graph)
    if save_pickle:
        pickle_fn = os.path.join(folder, "pickle", graph_name+".pickle")
        print(f"Saving {pickle_fn}")
        with open(pickle_fn, "wb") as pickle_file:
            pickle.dump(graph, pickle_file)
    if save_csv:
        csv_fn = os.path.join(folder, "csv", graph_name+".csv")
        print(f"Saving {csv_fn}")
        to_csv(graph, csv_fn)
    if save_turtle:
        turtle_fn = os.path.join(folder, "ttl", graph_name+".ttl")
        print(f"Saving {turtle_fn}")
        graph.serialize(turtle_fn)

def to_csv(graph, filename):
    results = graph.query("select ?s ?p ?o where { ?s ?p ?o } order by ?p ?s ?o")
    namespace = graph.namespace_manager
    triples = []
    for res in results:
        s, p, o = res['s'], res['p'], res['o']
        s, p = s.n3(namespace), p.n3(namespace)
        if not isinstance(o, Literal):
            o = o.n3(namespace)
        triples.append((s, p, o))
    df = pd.DataFrame(triples, columns=("subject", "predicate", "object"))
    df.to_csv(filename, index=False)

def n3_to_RDFLib(s):
    "Convert a string/n3 representation of an RDF term to an RDFLib RDF term"
    s_no_whitespace = re.sub(r"\s+", "_", s)
    try:
        if string_is_URI(s):
            prefix, value = split_uri(s)
            return prefix[value]
        elif string_is_URI(s_no_whitespace):
            prefix, value = split_uri(s_no_whitespace)
            return prefix[value]
        out = from_n3(s, nsm=NAMESPACE_MANAGER)
        if isinstance(out, BNode):
            # Avoid blank nodes
            return Literal(s)
        return out
    except Exception as e:
        # print(f"WARNING: string {s} cannot be converted to URIRef. {e}")
        # return something that does not make sense, to avoid matching anything
        return HKG["qwertyuiopasdfghjk"]

def load_pickled_graph(graph_filename):
    with open(graph_filename, "rb") as p:
        graph = pickle.load(p)
    return graph

def load_named_graph(graph_name):
    """
    Load one of the following graphs:

     - "main_graph"
     - "full_graph" = "main_graph" + all drugs and diseases of DBpedia
     - "full_graph_abstract" = "full_graph" + LLM-generated triples from DBpedia abstracts
    """
    file_path = os.path.realpath(__file__)
    hkg_folder = os.path.dirname(file_path)
    project_folder = os.path.dirname(hkg_folder)
    pickle_data_folder = os.path.join(project_folder, "data/graphs/pickle")
    main_graph_fn = os.path.join(pickle_data_folder, "main_graph.pickle")
    all_diseases_fn = os.path.join(pickle_data_folder, "all_diseases.pickle")
    all_drugs_fn = os.path.join(pickle_data_folder, "all_drugs.pickle")
    full_graph_abstract_fn = os.path.join(pickle_data_folder, "full_graph_abstract_triples_filtered.pickle")
    if graph_name == "main_graph":
        graph = load_pickled_graph(main_graph_fn)
    elif graph_name == "full_graph":
        graph = load_pickled_graph(main_graph_fn) + \
                    load_pickled_graph(all_diseases_fn) + \
                    load_pickled_graph(all_drugs_fn)
    elif graph_name == "full_graph_abstract":
        graph = load_pickled_graph(main_graph_fn) + \
                    load_pickled_graph(all_diseases_fn) + \
                    load_pickled_graph(all_drugs_fn) + \
                    load_pickled_graph(full_graph_abstract_fn)
    else:
        raise ValueError(f"Unkown graph name: {graph_name}")
    return graph
