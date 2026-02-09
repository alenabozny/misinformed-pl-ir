#!/usr/bin/env python3

import datetime
import glob
import os
import pandas as pd
import subprocess

from fastwarc.warc import ArchiveIterator
from multiprocessing import Pool

warc_files = glob.glob("./data/cc-news-2020/01/*.wet.gz")
warc_files.extend(glob.glob("./data/cc-news-2020/02/*.wet.gz"))
warc_files.extend(glob.glob("./data/cc-news-2020/03/*.wet.gz"))
warc_files.extend(glob.glob("./data/cc-news-2020/04/*.wet.gz"))

run_files = glob.glob("./misinfo-runs/adhoc/2020/*")
run_urns = set()
# Get necessary doc ids
for run in run_files:
    colnames = ["qid", "Q0", "docid", "rank", "score", "STANDARD"]
    run_df = pd.read_csv(run, sep=" ", header=None, names=colnames)
    run_df["docid_urn"] = run_df.docid.apply(lambda s: "<urn:uuid:"+s+">")
    run_urns = run_urns.union(set(run_df.docid_urn.unique()))

for i in [1,2,3,4]:
    os.makedirs(f"./csv/cc-news-2020/0{i}", exist_ok=True)

def process(file):
    basename = os.path.basename(file)
    csv_dirname = os.path.join("./csv/", os.path.dirname(file).replace("data/", ""))
    csv_name = basename.split('.')[0]+".csv"
    csv_temp = basename.split('.')[0]+".temp"
    csv_name = os.path.join(csv_dirname, csv_name)
    csv_temp = os.path.join(csv_dirname, csv_temp)
    if not os.path.exists(csv_name):
        temp = []
        with open(file, 'rb') as f:
            for record in ArchiveIterator(f):
                if "WARC-Refers-To" in record.headers:
                    id_ = record.headers['WARC-Refers-To']
                    if id_ in run_urns:
                        body = record.reader.read().decode()
                        temp.append((id_, body))
        df = pd.DataFrame(temp)
        df.to_csv(csv_temp, header=False, sep=",", index=False)
        os.rename(csv_temp, csv_name)

print("Obtaining retrieved TREC 2020: one .csv file for each .gz")
before = datetime.datetime.now()
with Pool(4) as pool:
    counts = pool.map(process, warc_files)
after = datetime.datetime.now()
print(f"Done in: {after-before}")


print("Joining the .csv files in a single file")
input_files = glob.glob("./csv/cc-news-2020/**/*.csv")
my_cmd = ['cat'] + input_files
with open('./data/trec2020_first_stage_retrieved_docs.csv', "w") as outfile:
    subprocess.run(my_cmd, stdout=outfile)
print("Done")
