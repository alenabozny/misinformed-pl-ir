#!/usr/bin/env python3

import json
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def doc_qid_correctness(doc_id, docs_correctness_dict):
    "Compute the correctness score for a document"
    # print(docs_correctness_dict['9947'])

    triples = docs_correctness_dict[str(doc_id)]["triples"]
    n_triples = len(triples)

    if n_triples == 0:
        return 0

    counts = {
        'TP': sum(1 for t in triples if t['binary_label'] == 'TP'),
        'FP': sum(1 for t in triples if t['binary_label'] == 'FP'),
        'TN': sum(1 for t in triples if t['binary_label'] == 'TN'),
        'FN': sum(1 for t in triples if t['binary_label'] == 'FN'),
        'UNKNOWN': sum(1 for t in triples if t['label'] == 'UNKNOWN'),
        'INVALID': sum(1 for t in triples if t['label'] == 'INVALID')
    }

    if n_triples == counts['INVALID']:
        # There are no valid triples
        return 0

    tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
    pos_accuracy = tp/(tp+fn) if (tp+fn) > 0 else 0
    neg_accuracy = tn/(tn+fp) if (tn+fp) > 0 else 0
    return pos_accuracy + neg_accuracy

def rerank_with_correctness(
    init_ranking_df: pd.DataFrame,
    docs_correctness_dict: dict,
    top_n: int,
    weight_topicality: float,
    weight_correctness: float,
    output_folder: str):
    "Re-rank documents combining the topicality and correctness scores."

    # Create output filename
    output_path = f"{output_folder}/reranked_top{top_n}_topicality{weight_topicality}_correctness{weight_correctness}.txt"

    scaler = MinMaxScaler()

    # Split into top-N and remaining documents upfront
    top_n_df = init_ranking_df.groupby('qid').head(top_n).copy()
    remaining_docs = init_ranking_df.groupby('qid').tail(-top_n).copy()

    # Merge top-N with correctness scores
    top_n_df["correctness_score"] = top_n_df.apply(
        lambda x: doc_qid_correctness(x.doc_id,
                                      docs_correctness_dict), axis=1)

    # Fill missing correctness scores with 0
    top_n_df['correctness_score'] = top_n_df['correctness_score'].fillna(0)

    # Normalize scores per query
    top_n_df['normalized_topical'] = top_n_df.groupby('qid')['score'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )

    top_n_df['normalized_correctness'] = top_n_df.groupby('qid')['correctness_score'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )

    # Calculate weighted score
    top_n_df['score'] = (weight_topicality * top_n_df['normalized_topical'] +
                        weight_correctness * top_n_df['normalized_correctness']) + 100

    # Sort reranked documents
    reranked_docs = top_n_df[['qid', 'doc_id', 'score']].sort_values(
        ['qid', 'score'], ascending=[True, False]
    )

    # Combine reranked and remaining documents
    combined_df = pd.concat([reranked_docs, remaining_docs], ignore_index=True)

    # Sort by qid and score
    combined_df = combined_df.sort_values(['qid', 'score'], ascending=[True, False])

    # Add ranks
    combined_df['rank'] = combined_df.groupby('qid').cumcount() + 1

    # Add runid
    combined_df['runid'] = f'reranked_top{top_n}_t{weight_topicality}_c{weight_correctness}'

    combined_df = combined_df.drop(['Q0'], axis=1, errors='ignore')
    combined_df['Q0'] = 'Q0'

    combined_df = combined_df[['qid','Q0' ,'doc_id', 'rank', 'score', 'runid']]

    combined_df['score'] = combined_df['score'].apply(lambda x: 0.0 if np.isclose(x, 0, atol=1e-10) else x)

    # Save results to file
    combined_df.to_csv(output_path, sep="\t", header=False, index=False)

def main(topicality, correctness, output):
    "Main function to run the reranking pipeline"

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Load initial rankings
    init_ranking = pd.read_csv(
        topicality,
        names=['qid','q0','doc_id','rank','score','runid'],
        sep=None,
        engine='python'
    )

    # Load correctness scores
    with open(correctness, "r") as f:
        correctness_list = json.load(f)
    docs_correctness_dict = {doc["doc_id"]: doc for doc in correctness_list}

    configs = []
    # depths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    depths = [200]
    weights = [(round(1-i*0.1, 1), round(i*0.1, 1)) for i in range(11)]

    for depth in depths:
        for weight_topicality, weight_correctness in weights:
            configs.append({
                'top_n': depth,
                'weight_topicality': weight_topicality,
                'weight_correctness': weight_correctness
            })

    # Run reranking for each configuration
    for config in configs:
        rerank_with_correctness(
            init_ranking_df=init_ranking,
            docs_correctness_dict=docs_correctness_dict,
            output_folder=output,
            **config
        )

    print(f"Reranking complete! Results saved in {output}")


if __name__ == "__main__":

    # custom
    topicality = "./data/custom_bm25_results.txt" #f"./misinfo-runs/adhoc/2020/run.misinfo-2020-description.bm25_en.txt"
    correctness = "./data/correctness_scores/custom_concat_full_graph_abstract_triple-level_correctness_scores.json" #f"./data/correctness_scores/trec2020_first_stage_retrieved_docs_passages_RDF_triples_full_graph_abstract_triple-level_correctness_scores_dense-10-match.json"
    output = f"./misinfo-runs/adhoc/custom/"
    print("Reranking custom...")
    main(topicality, correctness, output)

    # # TREC 2021
    # topicality = f"./misinfo-runs/adhoc/2021/run.misinfo-2021-description.bm25.txt"
    # correctness = f"./data/correctness_scores/trec2021_22_first_stage_retrieved_docs_passages_RDF_triples_full_graph_abstract_triple-level_correctness_scores_dense-10-match.json"
    # output = f"./misinfo-runs/adhoc/2021/"
    # print("Reranking TREC 2021...")
    # main(topicality, correctness, output)

    # # TREC 2022
    # topicality = f"./misinfo-runs/adhoc/2022/run.misinfo-2022-question.bm25.txt"
    # correctness = f"./data/correctness_scores/trec2021_22_first_stage_retrieved_docs_passages_RDF_triples_full_graph_abstract_triple-level_correctness_scores_dense-10-match.json"
    # output = f"./misinfo-runs/adhoc/2022/"
    # print("Reranking TREC 2022...")
    # main(topicality, correctness, output)
