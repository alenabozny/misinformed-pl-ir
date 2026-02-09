#!/usr/bin/env python3

import json
import pandas as pd
import re

from hkg.graph_utils import load_named_graph, DBO


main_wl = load_named_graph("main_graph")

abstracts = set(main_wl.subject_objects(predicate=DBO.abstract))
abstracts = { (s.n3(main_wl.namespace_manager), \
               re.sub(r"https?://dbpedia.org/\w+/", "", s).replace("_", " ").lower(), \
               o) for s, o in abstracts}
abstracts = sorted(abstracts)

df = pd.DataFrame(abstracts, columns=["URI", "Label", "Abstract"])

jsonl_name = "./data/main_graph_abstracts.jsonl"
print(f"Saving {jsonl_name}...")
with open(jsonl_name, "w") as j:
    for uri, label, abstract in abstracts:
        d = {"uri": uri, "label":label, "abstract": abstract}
        j.write(json.dumps(d)+"\n")
print("Done.")
