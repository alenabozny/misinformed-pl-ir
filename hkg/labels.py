#!/usr/bin/env python3

import re
import networkx as nx
import functools

from collections import defaultdict
from rdflib import RDFS, OWL, URIRef, Literal, Graph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from tqdm import tqdm
from urllib.parse import unquote

from retriv import DenseRetriever

from hkg.graph_utils import HKG, n3_to_RDFLib, subgraph
from hkg.namespaces import DBR
from hkg.rdf_regex import (is_camel_case, to_snake_case, extract_value, string_is_URI,
                       split_uri, parse_triple, camel_case_split)

POSITIVE_PREDICATES = [HKG.treatment, HKG.medication, HKG.prevention]
NEGATIVE_PREDICATES = [HKG.complication, HKG.cause, HKG.symptom, HKG.risk]
HKG_PREDICATES = POSITIVE_PREDICATES + NEGATIVE_PREDICATES + [HKG.diagnosis]
SIMILAR_PREDICATES = { p: set() for p in HKG_PREDICATES }
SIMILAR_PREDICATES[HKG.treatment].add(HKG.medication)
SIMILAR_PREDICATES[HKG.medication].add(HKG.treatment)
SIMILAR_PREDICATES[HKG.complication].add(HKG.cause)
SIMILAR_PREDICATES[HKG.cause].add(HKG.complication)


def strip_quotes(text):
    return re.sub(r'^("|\')+|("|\')+$', "", text.strip())

def URI2labels(uri):
    if isinstance(uri, str):
        uri = n3_to_RDFLib(uri)
    uri = unquote(uri)
    labels = set()
    noprefix = str(extract_value(uri)).replace("_", " ")
    labels.add(noprefix.lower())
    labels.add(noprefix.lower().replace("-", " "))
    if is_camel_case(noprefix):
        noprefix = to_snake_case(noprefix, lowercase=True, capitalize_first=False).replace("_", " ")
        labels.add(noprefix)
        labels.add(noprefix.replace("-", " "))
    if re.search(r"\(\w+\)", uri):
        for label in labels.copy():
            before_bracket = label.split("(")[0]
            after_bracket = label.split("(")[-1]
            main_label = before_bracket.strip()
            specifier = after_bracket.replace(")", "").strip()
            labels.add(main_label)
            labels.add(specifier+" "+main_label)
    labels.discard("")
    return labels


class Labels:
    """
    A class that takes care of matching terms and triples with those in a given RDF graph.

    The main methods are `verify_triple` and `verify_triple_two_steps`.
    """
    # TODO: the name `Labels` comes from an older class.
    # It should be renamed to something like `TripleMatcher`
    def __init__(self, graph, dense_index=None):
        print("Processing graph")
        graph_normalized = Graph()
        for s, p, o in graph:
            # Remove information about language and datatype form literals
            # to make matching easier
            if isinstance(o, Literal):
                graph_normalized.add((s, p, Literal(o.value)))
            else:
                graph_normalized.add((s, p, o))
        self.graph = graph_normalized
        self.dense_retriever = None
        if dense_index is not None:
            print(f"Loading dense index {dense_index}")
            self.dense_retriever = DenseRetriever.load(dense_index)

        self.uri_to_labels = defaultdict(set) # URI -> string
        self.label_to_uris = defaultdict(set) # string -> URI
        self.hkg_entities = set()

        print("Computing equivalence classes")
        nx_sameas = rdflib_to_networkx_graph(subgraph(self.graph, "owl:sameAs"))
        self.uri_representative = dict()
        self.same_uris = dict()
        for component in nx.connected_components(nx_sameas):
            rep = next(iter(component))
            self.uri_representative[rep] = rep
            self.same_uris[rep] = component
            for c in component:
                self.uri_representative[c] = rep

        for predicate in HKG_PREDICATES:
            for s, o in self.graph.subject_objects(predicate=predicate):
                self.hkg_entities.add(s)
                self.hkg_entities.add(o)

        self.predicates = set(self.graph.predicates())

        print("Collecting entities for which to store labels")
        entities = set()
        relevant_predicates = self.predicates
        relevant_predicates.discard(OWL.sameAs)
        for pred in relevant_predicates:
            entities = entities.union(self.graph.subjects(predicate=pred))
            entities = entities.union(self.graph.objects(predicate=pred))
        entities = {ent for ent in entities if isinstance(ent, URIRef)}

        print("Computing labels")
        for entity in tqdm(sorted(entities)):
            self.__update_labels(entity)

        self.uri_labels = set(self.label_to_uris.keys()) # set of strings
        validate_label = lambda x: isinstance(x, Literal) and not isinstance(x.value, int) and x.value is not None
        self.other_literals = set(obj.value for obj in self.graph.objects() if validate_label(obj))
        self.other_literals = self.other_literals.difference(self.uri_labels)
        print("Done.")

    def sameas(self, uri):
        if uri in self.uri_representative.keys():
            return self.same_uris[self.uri_representative[uri]]
        return {uri}

    def _validate_predicate(self, pred):
        rdf_pred = n3_to_RDFLib(pred)
        if rdf_pred in self.predicates:
            return True, rdf_pred
        # try to recover an allowed predicate
        if string_is_URI(pred):
            # try to cover cases like dbp:medication
            # by converting to hkg:medication
            _, value = split_uri(pred)
            if (rdf_pred := HKG[strip_quotes(value)]) in self.predicates:
                return True, rdf_pred
        if (rdf_pred := HKG[strip_quotes(pred)]) in self.predicates:
            return True, rdf_pred
        return False, pred

    @functools.cache
    def dense_match(self, term, cutoff=10):
        if self.dense_retriever is None:
            print("Dense retriever is not loaded")
            return set()
        if string_is_URI(term):
            term = extract_value(term)
            term = term.replace("_", " ")
            if is_camel_case(term):
                term = " ".join(camel_case_split(term))
        results = self.dense_retriever.search(query=term, return_docs=True, cutoff=cutoff)
        return {result['text'] for result in results}

    @functools.cache
    def _match_triple(self, triple: str, binary=True, dense=False, cutoff=10):
        matched_triples = set()
        match = parse_triple(triple)
        # Check if the input matches the triple pattern
        if not match:
            if binary:
                return "TRUE", matched_triples
            else:
                return "INVALID", matched_triples
        # Check that the predicate is valid
        pred = match.group("predicate")
        valid_pred, rdf_pred = self._validate_predicate(pred)
        if not valid_pred:
            if binary:
                return "TRUE", matched_triples
            else:
                return "INVALID", matched_triples
        # Check whether the "predicted" value is "ASK"
        predicted = match.group("truth")
        if not binary and predicted == "ASK":
            return "INVALID", matched_triples
        # Check wehther the subject matches any URI in the graph
        sub = match.group("subject")
        sub_match = { sm for sm in self.match_term(sub, check_hkg=True) if isinstance(sm, URIRef) }
        if dense:
            for dense_match in self.dense_match(sub, cutoff=cutoff):
                sub_match.update(self.match_term(dense_match, check_hkg=True))
            sub_match = { sm for sm in sub_match if isinstance(sm, URIRef) }
        if not sub_match and not binary:
            return "UNKNOWN", matched_triples
        # Check wehther the object matches any URI in the graph
        obj = match.group("object")
        obj_match = self.match_term(obj, check_hkg=True)
        if dense:
            for dense_match in self.dense_match(obj, cutoff=cutoff):
                obj_match.update(self.match_term(dense_match, check_hkg=True))
        if not obj_match and not binary:
            return "UNKNOWN", matched_triples
        # return matches
        for s in sub_match:
            for o in obj_match:
                matched_triples.add((s, rdf_pred, o))
        return predicted, matched_triples

    @functools.cache
    def _similar_triple_in_graph(self, triple):
        """
        Returns True if a triple similar to `triple` is in the graph

        "Similar" in this method refers to whether there is a triple with a predicate in
        SIMILAR_PREDICATES, it does not mean semantic similarity (computed with a dense
        index).

        The function does not consider the original triple: if the original triple is in
        the graph, and no similar triple is in the graph, the output will be False.
        """
        s, p, o = triple
        for similar_p in SIMILAR_PREDICATES.get(p, p):
            if (s, similar_p, o) in self.graph:
                return True
        return False

    @functools.cache
    def _inverse_triple_in_graph(self, triple):
        "Returns True if the inverse of `triple` is the graph"
        s, p, o = triple
        return  (o, p, s) in self.graph

    @functools.cache
    def triple_in_graph(self, triple: tuple, allow_similar_predicates=False, allow_inverse=False):
        if triple in self.graph:
            return True
        if allow_inverse:
            if self._inverse_triple_in_graph(triple):
                return True
        if allow_similar_predicates:
            if self._similar_triple_in_graph(triple):
                return True
        if allow_similar_predicates and allow_inverse:
            s, p, o = triple
            if self._similar_triple_in_graph((o, p, s)):
                return True
        return False

    @functools.cache
    def verify_triple(self, triple: str, binary=True,
                      allow_similar_predicates=False,
                      allow_inverse=False, dense=False, cutoff=10):
        """
        Input: triple associated with a "truth degree" = "TRUE", "FALSE", "ASK"
        E.g.: `TRUE ( dbr:Migraine hkg:medication dbr:Paracetamol )`

        Meaning of input:
         - TRUE ( s p o ) asserts that the triple (s p o) is True
         - FALSE ( s p o ) asserts that the triple (s p o) is False
         - ASK ( s p o ) asks whether the triple (s p o) is True

        Output (binary case) for triple = `TRUTH (s p o)`:
         - If (s, p, o) IN self.graph (i.e., it's True):
            - `TRUE  ( s, p, o )` is verified as True: True Postiive
            - `FALSE ( s, p, o )` is verified as False: False Negative
         - If (s, p, o) NOT IN self.graph (i.e., it's False),
            - `TRUE  ( s, p, o )` is verified as False: False Positive
            - `FALSE ( s, p, o )` is verified as True: True Negative

        Note: ASK behaves like TRUE in the binary setting.

        output (non-binary case) for triple = `TRUTH (s p o)`:
            - If `triple` does not match the triple pattern:
                - return INVALID
            - Else if `p` is not a predicate of self.graph:
                - return INVALID
            - Else if TRUTH == ASK:
                - return INVALID
            - Else if `s` does not match any term in self.graph:
                - return UNKNOWN
            - Else if `o` does not match any term in self.graph:
                - return UNKNOWN
            - Else:
                - binary case
        """
        predicted_truth, matched_triples = self._match_triple(triple, binary, dense=dense,
                                                              cutoff=cutoff)
        if not binary and predicted_truth in { "INVALID", "UNKNOWN" }:
            return predicted_truth
        return self._verify_triple_binary(predicted_truth, matched_triples,
                                          allow_similar_predicates,
                                          allow_inverse)

    @functools.cache
    def verify_triple_two_steps(self, triple: str, binary=True,
                                allow_similar_predicates=False, allow_inverse=False, cutoff=10):
        if binary:
            return self._verify_triple_two_steps_binary(triple,
                                                        allow_similar_predicates=allow_similar_predicates,
                                                        allow_inverse=allow_inverse, cutoff=cutoff)
        return self._verify_triple_two_steps_nonbinary(triple,
                                                       allow_similar_predicates=allow_similar_predicates,
                                                       allow_inverse=allow_inverse, cutoff=cutoff)

    @functools.cache
    def _verify_triple_two_steps_binary(self, triple: str, allow_similar_predicates=False,
                                        allow_inverse=False, cutoff=10):
        match = parse_triple(triple)
        if not match:
            raise ValueError(f"Not a triple: {triple}")
        predicted_truth = match.group("truth")
        # First, try verifying without dense
        output = self.verify_triple(triple, binary=True, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse)
        # In case of false positives, try dense
        if predicted_truth == "TRUE" and not output:
            # If the predicted_truth is TRUE and the output is False, it means it's a false positive
            output = self.verify_triple(triple, binary=True, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse, dense=True, cutoff=cutoff)
        # In case of true negatives, try dense
        if predicted_truth == "FALSE" and output:
            # If the predicted truth is FALSE and the output is True, it means it's a true negative
            output = self.verify_triple(triple, binary=True, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse, dense=True, cutoff=cutoff)
        return output

    @functools.cache
    def _verify_triple_two_steps_nonbinary(self, triple: str,
                                           allow_similar_predicates=False,
                                           allow_inverse=False, cutoff=10):
        match = parse_triple(triple)
        if not match:
            raise ValueError(f"Not a triple: {triple}")
        predicted_truth = match.group("truth")
        # First, try verifying without dense
        output = self.verify_triple(triple, binary=False, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse)
        # Unkown output
        if output == "UNKNOWN":
            output = self.verify_triple(triple, binary=False, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse, dense=True, cutoff=cutoff)
        # In case of false positives, try dense
        if predicted_truth == "TRUE" and not output:
            # If the predicted_truth is TRUE and the output is False, it means it's a false positive
            output = self.verify_triple(triple, binary=False, allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse, dense=True, cutoff=cutoff)
        # In case of true negatives, try dense
        if predicted_truth == "FALSE" and output:
            # If the predicted truth is FALSE and the output is True, it means it's a true negative
            output = self.verify_triple(triple, binary=False,allow_similar_predicates=allow_similar_predicates, allow_inverse=allow_inverse, dense=True, cutoff=cutoff)
        return output

    def _verify_triple_binary(self, predicted_truth, matched_triples,
                              allow_similar_predicates=False,
                              allow_inverse=False):
        if predicted_truth == "TRUE" or predicted_truth == "ASK":
            prediction = True
        else: # predicted_truth == "False"
             prediction = False
        if not matched_triples:
            # If there are no matched triples in the graph, the input triple is False
            # If prediction = True -> return False,
            # If prediction = False -> return True. I.e.:
            return not prediction
        for rdf_triple in matched_triples:
            if self.triple_in_graph(rdf_triple, allow_similar_predicates, allow_inverse):
                # I.e., the input triple is True
                # if prediction = True, return True,
                # if prediction = False return False. I.e.:
                return prediction
        # No matches found: the input triple is False
        # if prediction = True, return False,
        # if prediction = False return True. I.e.:
        return not prediction

    def match_term_two_steps(self, term, check_hkg=True, cutoff=10):
        matches = self.match_term(term, check_hkg=check_hkg)
        if not matches:
            dense_matches = self.dense_match(term, cutoff=cutoff)
            for dense_match in dense_matches:
                matches.update(self.match_term(dense_match, check_hkg=check_hkg))
        return matches

    def match_term(self, term, check_hkg=True):

        def process_uri(term):
            matches = self.match_uri(term, check_hkg=check_hkg)
            prefix, value = split_uri(term)
            if not value[0].isupper():
                value = value[0].upper() + value[1]
                matches.update(self.match_uri(prefix[value], check_hkg=check_hkg))
            for label in URI2labels(term):
                matches.update(self.match_literal(label, check_hkg=check_hkg))
            return matches

        def process_literal(term):
            term_variants = set([term, term.lower(), strip_quotes(term),
                                 strip_quotes(term.lower())])
            matches = set()
            for variant in term_variants:
                matches.update(self.match_literal(variant, check_hkg=check_hkg))

            urified_term = re.sub(r"\s+", "_", term)
            urified_term = strip_quotes(urified_term[0].upper() + urified_term[1:])
            if string_is_URI(DBR[urified_term]):
                matches.update(process_uri(DBR[urified_term]))

            urified_term = urified_term[0] + urified_term[1:].lower()
            if string_is_URI(DBR[urified_term]):
                matches.update(process_uri(DBR[urified_term]))

            return matches

        term = term.strip()
        term = re.sub(r"\s+", " ", term)
        if isinstance(term, URIRef):
            return process_uri(term)
        elif isinstance(term, str):
            if string_is_URI(term):
                return process_uri(term)
            if string_is_URI(term_nospace := re.sub(r"\s+", "_",  term)):
                return process_uri(term_nospace)
            else:
                return process_literal(term)
        elif isinstance(term, Literal):
            return process_literal(term.value)
        else:
            print(f"Warning: unexpected type {type(term)} for term {term}")
        return set()

    def match_uri(self, uri, check_hkg=True):
        matches = set()
        if isinstance(uri, str):
            uri = n3_to_RDFLib(uri)
        if uri in self.uri_to_labels:
            if check_hkg:
                if uri in self.hkg_entities:
                    matches.add(uri)
            else:
                matches.add(uri)
            for l in self.uri_to_labels[uri]:
                matches.update(self.match_literal(l, check_hkg=check_hkg))
        return matches

    def match_literal(self, literal, check_hkg=True):
        # input `literal` is a string
        matches = set()
        ll = Literal(literal)
        if literal in self.uri_labels:
            if check_hkg:
                if ll in self.hkg_entities:
                    matches.add(ll)
            else:
                matches.add(ll)
            for uri in self.label_to_uris[literal]:
                for same in self.sameas(uri):
                    if check_hkg:
                        if same in self.hkg_entities:
                            matches.add(same)
                    else:
                        matches.add(same)
        if literal in self.other_literals:
            if check_hkg:
                if ll in self.hkg_entities:
                    matches.add(ll)
            else:
                matches.add(ll)
        return matches

    def __update_labels(self, entity):
        self.uri_to_labels[entity].update(URI2labels(entity))
        rdfs_labels = set(self.graph.objects(subject=entity, predicate=RDFS.label))
        rdfs_labels =  {re.sub(r"\s+", " ", lab.lower()) for lab in rdfs_labels if lab}
        self.uri_to_labels[entity].update(rdfs_labels)
        sameas = set(self.graph.objects(subject=entity, predicate=OWL.sameAs))
        for same in sameas:
            self.uri_to_labels[entity].update(URI2labels(same))
            labels = set(self.graph.objects(subject=same, predicate=RDFS.label))
            labels =  {re.sub(r"\s+", " ", lab.lower()) for lab in labels if lab}
            self.uri_to_labels[entity].update(labels)
        for label in self.uri_to_labels[entity]:
            self.label_to_uris[label].add(entity)

    def __get_labels(self, entity):
        if isinstance(entity, URIRef):
            return self.uri_to_labels[entity]
        else:
            return {entity}

    def __iter__(self):
        return iter(self.uri_to_labels.keys())

    def __contains__(self, obj):
        return obj in self.uri_to_labels.keys()

    def __getitem__(self, item):
        return self.__get_labels(item)

if __name__ == "__main__":

    from hkg.graph_utils import load_named_graph

    graph = load_named_graph("main_graph")
    labels = Labels(graph)

    print(labels.match_term("Vitamin C"))
    # Output: {rdflib.term.URIRef('http://dbpedia.org/resource/Vitamin_C'), rdflib.term.Literal('vitamin c')}

    print(labels.verify_triple('TRUE ( dbr:Migraine hkg:medication dbr:Paracetamol )'))
    # Output: True

    print(labels.verify_triple('TRUE ( dbr:Migraine hkg:medication Paracetamol )'))
    # Output: True

    print(labels.verify_triple('TRUE ( dbr:Migraine hkg:medication dbr:Vitamin_C )'))
    # Output: False
