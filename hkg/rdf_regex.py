#!/usr/bin/env python3

import re
from rdflib import RDF, RDFS, OWL

from hkg.namespaces import PREFIX_TO_NAMESPACE, DBR


CAMELCASE = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

def is_camel_case(string):
    return bool(CAMELCASE.match(string))

def camel_case_split(text):
    matches = CAMELCASE.finditer(text)
    return [m.group(0) for m in matches]

def to_snake_case(text, lowercase=True, capitalize_first=False):
    matches = camel_case_split(text)
    if lowercase:
        snake = "_".join(m.lower() for m in matches)
    else:
        snake = "_".join(matches)
    if capitalize_first:
        snake = snake[0].upper()+snake[1:]
    return snake


DBR_PATTERN = 'https?://dbpedia.org/resource/'
DBO_PATTERN = 'https?://dbpedia.org/ontology/'
DBP_PATTERN = 'https?://dbpedia.org/property/'
HKG_PATTERN = 'https?://example.org/health_kg/'

LONG_PREFIX = re.compile("|".join([DBP_PATTERN, DBR_PATTERN, DBO_PATTERN, HKG_PATTERN, str(RDF), str(RDFS), str(OWL)]))
SHORT_PREFIX = re.compile("|".join(["dbp:", "dbr:", "dbo:", "hkg:", "rdf:", "rdfs:", "owl:", "dkg:"]))
PREFIXED_URI_PATTERN = re.compile(rf"(?P<prefix>{SHORT_PREFIX.pattern})(?P<value>\S+)")
LONG_URI_PATTERN = re.compile(rf"(?P<prefix>{LONG_PREFIX.pattern})(?P<value>\S+)")
URI_PATTERN = re.compile(rf"<?(?P<prefix>({SHORT_PREFIX.pattern}|{LONG_PREFIX.pattern}))(?P<value>[^\s>]+)>?")
SINGLE_URI_PATTERN = re.compile(rf"^{URI_PATTERN.pattern}$")
SINGLE_HKG_URI_PATTERN = re.compile(r"^(?P<prefix>hkg):(?P<value>\S+)$")

def parse_uris_iter(text):
    "Iterate over URIs in text"
    yield from URI_PATTERN.finditer(text.strip())

def string_is_URI(text):
    "Is 'text' an URI?"
    return bool(SINGLE_URI_PATTERN.match(text))

def split_uri(uri):
    if match := URI_PATTERN.match(uri):
        prefix = match.group("prefix")
        value = match.group("value")
        return PREFIX_TO_NAMESPACE.get(prefix.strip(":"), DBR), value
    else:
        raise ValueError(f"Not a uri: {uri}")

def extract_value(uri):
    if match := URI_PATTERN.match(uri):
        return match.group("value")
    else:
        raise ValueError("Not a uri")

def extract_prefix(uri):
    if match := URI_PATTERN.match(uri):
        prefix = match.group("prefix")
        return PREFIX_TO_NAMESPACE[prefix.strip(":")]
    else:
        raise ValueError("Not a uri")

TRIPLE_PATTERN = re.compile(r"(?P<truth>TRUE|FALSE|ASK)\s*\(\s*(?P<subject>\S+:?[^:]+)\s+(?P<predicate>\S+:\S+)\s+(?P<object>[\"']?\S+(\s+(\w|[!?,'-.])+)*[\"']?)\s*\)")

def parse_triple(text):
    "Tries to match 'text' with a triple pattern"
    return TRIPLE_PATTERN.match(text.strip())

def parse_triples_iter(text):
    "Iter over all triples in text"
    yield from TRIPLE_PATTERN.finditer(text.strip())

if __name__ == "__main__":

    from graph_utils import DBO

    uris = [
            "<http://dbpedia.org/resource/Graves'_disease>",
            "http://dbpedia.org/resource/Graves'_disease",
            DBO["Graves'_disease"],
            "http://dbpedia.org/resource/Graves_disease",
            "<http://dbpedia.org/resource/Graves_(disease)>",
            ]

    for i, uri in enumerate(uris):
        print(f"{i, uri = }")
        value = extract_value(uri)
        print(f"{value = }")
        prefix = extract_prefix(uri)
        print(f"{prefix = }")
        print()
