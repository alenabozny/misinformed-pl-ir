#!/usr/bin/env python3

import os

from utils import load_top_n_docs


def save_top_doc_ids_per_query(run_file, out_folder):
    run_basename = os.path.basename(run_file).replace(".txt", "")
    out_folder = os.path.join(out_folder, run_basename)
    os.makedirs(out_folder, exist_ok=True)

    qid_to_docs = load_top_n_docs(run_file)

    for qid, docids in qid_to_docs.items():
        out_file = os.path.join(out_folder, f"query_{qid}_top_docs.txt")
        with open(out_file, "w") as f:
            for i, docid in enumerate(docids):
                f.write(f"{i+1},{docid}\n")


if __name__ == "__main__":

    # for run_file in [
    #         "./misinfo-runs/adhoc/2020/run.misinfo-2020-description.bm25_en.txt",
    #         "./misinfo-runs/adhoc/2020/run.misinfo-2020-title.bm25_en.txt"
    #         ]:
    #     out_folder = "./data/top_docs/2020/"
    #     save_top_doc_ids_per_query(run_file, out_folder)

    # for run_file in [
    #         "./misinfo-runs/adhoc/2021/run.misinfo-2021-query.bm25.txt",
    #         "./misinfo-runs/adhoc/2021/run.misinfo-2021-description.bm25.txt"
    #         ]:
    #     out_folder = "./data/top_docs/2021/"
    #     save_top_doc_ids_per_query(run_file, out_folder)

    # for run_file in [
    #         "./misinfo-runs/adhoc/2022/run.misinfo-2022-query.bm25.txt",
    #         "./misinfo-runs/adhoc/2022/run.misinfo-2022-question.bm25.txt"
    #         ]:
    #     out_folder = "./data/top_docs/2022/"
    #     save_top_doc_ids_per_query(run_file, out_folder)

    out_folder = "./data/top_docs/custom/"
    save_top_doc_ids_per_query("./data/custom_bm25_results.tsv", out_folder)

    
