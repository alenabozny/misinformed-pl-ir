#!/usr/bin/env python3

import pandas as pd


def load_top_n_docs(run_file, n=1000):
    """
    Load the top N documents for each query from a result file.

    Args:
    run_file (str): Path to the result file.
    n (int): Number of top documents to load for each query.

    Returns:
    dict: A dictionary with query IDs as keys and lists of top document IDs as values.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(run_file, sep='\s+', header=None,
                     names=['qid', 'q0', 'doc_id', 'rank', 'score', 'run_name'])

    # Sort by query ID and rank to ensure correct ordering
    df = df.sort_values(['qid', 'rank'])

    # Group by query ID and get the top N documents for each query
    top_n_docs = df.groupby('qid').head(n)

    # Convert to a dictionary format
    result = top_n_docs.groupby('qid')['doc_id'].apply(list).to_dict()

    # Print summary of loaded documents
    return result
