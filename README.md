# Reproduced study "Fact-Driven Health Information Retrieval: Integrating LLMs and Knowledge Graphs to Combat Misinformation (ECIR 2025)"  on translated Polish MisiforMED-PL dataset

## This is a link to [the original experiment](https://github.com/GianCarloMilanese/fact-driven-health-ir)

This repository contains the cloned code related to the ECIR 2025 short paper
'Fact-Driven Health Information Retrieval: Integrating LLMs and Knowledge Graphs
to Combat Misinformation' extended with custom scripts and data, namely:

- ./queries/custom-queries.tsv
- ./trec-misinfo-resources/custom/*
- ./3_1_2_sentence_triplets_to_passages.py
- ./3_2_2_compute_bm25_for_custom.ipynb
- ./3_4_2_concatenate_RDF_jsons_into_jsonl.py
- ./3_6_evaluation.py

Running the scripts according to the instruction from the original dataset, with custom helper scripts run in the middle, will result in full reproduction of our experiments.
