import json
import pandas as pd

file_credible = "./trec-misinfo-resources/custom/cred_cred_en.csv"      # file where rows are credible
file_noncredible = "./trec-misinfo-resources/custom/noncred_noncred_en.csv"   # file where rows are non-credible

# adjust read function if not CSV, e.g. pd.read_json(...)
df_a = pd.read_csv(file_credible)
df_b = pd.read_csv(file_noncredible)

# Ensure required columns exist
required = {"triplet_r1", "EN_translation"}
for df, name in [(df_a, "file_credible"), (df_b, "file_noncredible")]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

# Add credibility labels
df_a = df_a.copy()
df_b = df_b.copy()
df_a["credibility"] = "credible"
df_b["credibility"] = "non-credible"

# Concatenate
df = pd.concat([df_a, df_b], ignore_index=True)

# Write JSONL in requested structure
out_path = "./data/custom_docs_passages.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        obj = {
            "doc_id": str(row["triplet_r1"]).strip(),
            "passages": {"passage_1": str(row["EN_translation"]).strip()},
            "credibility": row["credibility"]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(df)} records to {out_path}")
