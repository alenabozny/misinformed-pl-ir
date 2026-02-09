# `hkg` usage example

After creating HKG (see the main README of this repository), the `hkg` package
can be used to verify claims against this KG.

```python
from hkg.graph_utils import load_named_graph
from hkg.labels import Labels

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
```
