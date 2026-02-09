#!/usr/bin/env python3

from rdflib import Namespace
from rdflib import RDF, RDFS, OWL

DBR = Namespace('http://dbpedia.org/resource/')
DBO = Namespace('http://dbpedia.org/ontology/')
DBP = Namespace('http://dbpedia.org/property/')
HKG = Namespace('https://example.org/health_kg/')

PREFIX_TO_NAMESPACE = {
    "dbr": DBR, "dbp": DBP, "dbo": DBO, "dbpedia": DBR,
    "rdf": RDF, "rdfs": RDFS, "owl": OWL,
    "hkg": HKG,
    'https://dbpedia.org/resource/': DBR,
    'https://dbpedia.org/ontology/': DBO,
    'https://dbpedia.org/property/': DBP,
    'https://example.org/health_kg/': HKG,
    # In case an rdflib Namespace is used as key
    DBR: DBR,
    DBP: DBP,
    DBO: DBO,
    RDF: RDF,
    RDFS: RDFS,
    OWL: OWL,
    HKG: HKG,
    str(DBR): DBR,
    str(DBP): DBP,
    str(DBO): DBO,
    str(RDF): RDF,
    str(RDFS): RDFS,
    str(OWL): OWL,
    str(HKG): HKG,
}
