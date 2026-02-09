import json
from collections import defaultdict
import numpy as np
from scipy.stats import ttest_rel

def load_rankings(path):
    rankings = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            qid = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            score = float(parts[4])

            rankings[qid].append({
                "doc_id": doc_id,
                "rank": rank,
                "score": score
            })

    # sort by rank
    for qid in rankings:
        rankings[qid] = sorted(rankings[qid], key=lambda x: x["rank"])

    return rankings

def weighted_cred_at_k(docs, k):
    return sum(
        credibility.get(d["doc_id"], 0.0)
        for d in docs[:k]
    )

def compute_metrics(rankings, cutoffs=(3, 5, 10)):
    per_query = defaultdict(dict)

    for qid, docs in rankings.items():
        for k in cutoffs:
            per_query[qid][f"wcred@{k}"] = weighted_cred_at_k(docs, k)

    return per_query

def sig_marker(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def perf_arrow(baseline, value):
    if value > baseline:
        return r"$\uparrow$"
    elif value < baseline:
        return r"$\downarrow$"
    else:
        return ""

def print_latex_table(cutoffs):
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"Metric & Topic_1.0_Corr_0.0 & Topic_0.5_Corr_0.5 & Topic_0.3_Corr_0.7\\")
    print(r"\hline")

    for k in cutoffs:
        key = f"wcred@{k}"
        r = results[key]

        A = r["A_mean"]

        print(
            f"{key} & "
            f"{A:.3f} & "
            f"{r['B_mean']:.3f}"
            f"{perf_arrow(A, r['B_mean'])}"
            f"{sig_marker(r['B_p'])} & "
            f"{r['C_mean']:.3f}"
            f"{perf_arrow(A, r['C_mean'])}"
            f"{sig_marker(r['C_p'])} & "
            # f"{r['D_mean']:.3f}"
            # f"{perf_arrow(A, r['D_mean'])}"
            # f"{sig_marker(r['D_p'])} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(
        r"\caption{Weighted credibility comparison across systems. "
        r"Arrows indicate performance relative to Topic_1.0_Corr_0.0. "
        r"Significance tested using paired t-tests against Topic_1.0_Corr_0.0. "
        r"$^*$ $p<0.05$, $^{**}$ $p<0.01$, $^{***}$ $p<0.001$.}"
    )
    print(r"\label{tab:system_comparison}")
    print(r"\end{table}")


# ---------- Load credibility ----------
credibility = {}

with open("./data/custom_docs_passages.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        credibility[obj["doc_id"]] = 1.0 if obj["credibility"] == "credible" else 0.0


# ---------- Load all systems ----------
sysA_rankings = load_rankings(
    "./misinfo-runs/adhoc/custom/reranked_top200_topicality1.0_correctness0.0.txt"
)  # Topic_1.0_Corr_0.0

sysB_rankings = load_rankings(
    "./misinfo-runs/adhoc/custom/reranked_top200_topicality0.5_correctness0.5.txt"
)  # Topic_0.5_Corr_0.5

sysC_rankings = load_rankings(
    "./misinfo-runs/adhoc/custom/reranked_top200_topicality0.3_correctness0.7.txt"
)  # Topic_0.3_Corr_0.7

# sysD_rankings = load_rankings(
#     "./misinfo-runs/adhoc/custom/reranked_top200_topicality0.0_correctness1.0.txt"
# )  # System D

cutoffs = (5, 10, 100)

sysA = compute_metrics(sysA_rankings, cutoffs)
sysB = compute_metrics(sysB_rankings, cutoffs)
sysC = compute_metrics(sysC_rankings, cutoffs)
# sysD = compute_metrics(sysD_rankings, cutoffs)

# ------ Aggregate + paired t-tests -------

results = {}

systems = {
    "A": sysA,
    "B": sysB,
    "C": sysC
    # "D": sysD
}

for k in cutoffs:
    key = f"wcred@{k}"

    # common query set
    common_qs = set(sysA) & set(sysB) & set(sysC)#& set(sysD)

    vals = {
        s: np.array([systems[s][q][key] for q in common_qs])
        for s in systems
    }

    results[key] = {
        "A_mean": vals["A"].mean(),
        "B_mean": vals["B"].mean(),
        "C_mean": vals["C"].mean(),
        # "D_mean": vals["D"].mean(),

        "B_p": ttest_rel(vals["A"], vals["B"]).pvalue,
        "C_p": ttest_rel(vals["A"], vals["C"]).pvalue,
        # "D_p": ttest_rel(vals["A"], vals["D"]).pvalue,
    }


print_latex_table(cutoffs)


