#!/usr/bin/env python3

import datetime
import glob
import gzip
import json
import os
import pandas as pd
import subprocess

from collections import defaultdict
from multiprocessing import Pool

c4_folder = "./data/c4/en.noclean/"
csv_folder = "./csv/c4/en.noclean/"
os.makedirs(csv_folder, exist_ok=True)


run_files = glob.glob("./misinfo-runs/adhoc/2021/*")
run_files.extend(glob.glob("./misinfo-runs/adhoc/2022/*"))
run_docids = defaultdict(set)
# Get necessary doc ids
for run in run_files:
    colnames = ["qid", "Q0", "docid", "rank", "score", "STANDARD"]
    run_df = pd.read_csv(run, sep=" ", header=None, names=colnames)
    docids = list(run_df.docid.values)
    for docid in docids:
        split = docid.split(".")
        doc_line_nr = split[-1]
        c4_docid = ".".join(split[:-1])
        run_docids[c4_docid].add(int(doc_line_nr))

def process(docid):
    line_nrs = run_docids[docid]
    basename = docid.replace("en.noclean.", "")+".json.gz"

    file = os.path.join(c4_folder, basename)

    csv_name = docid.replace("en.noclean.", "")+".csv"
    csv_temp = docid.replace("en.noclean.", "")+".temp"
    csv_name = os.path.join(csv_folder, csv_name)
    csv_temp = os.path.join(csv_folder, csv_temp)
    if not os.path.exists(csv_name):
        temp = []
        with gzip.open(file, "r") as f:
            for i, line in enumerate(f):
                if i in line_nrs:
                    d = json.loads(line.strip())
                    text = d['text']
                    url = d['url']
                    timestamp = d['timestamp']
                    temp.append((f"{docid}.{i}",text,url,timestamp))
        df = pd.DataFrame(temp)
        df.to_csv(csv_temp, header=False, sep=",", index=False)
        os.rename(csv_temp, csv_name)

print("Obtaining retrieved TREC 2021-22 documents: one .csv file for each .gz")
before = datetime.datetime.now()
with Pool(4) as pool:
    pool.map(process, list(run_docids.keys()))
after = datetime.datetime.now()
print(f"Done in: {after-before}")


print("Joining the .csv files in a single file")
input_files = glob.glob("./csv/c4/en.noclean/*.csv")
my_cmd = ['cat'] + input_files
with open('./data/trec2021-22_first_stage_retrieved_docs.csv', "w") as outfile:
    subprocess.run(my_cmd, stdout=outfile)
print("Done")
