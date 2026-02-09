#!/usr/bin/env python3
"""
Computes a whitelist of English documents in the TREC 2020 HM collection.
"""

import datetime
import glob
import os
import subprocess

from fastwarc.warc import ArchiveIterator
from ftlangdetect import detect
from multiprocessing import Pool

# Suppress fasttext warnings
import fasttext
fasttext.FastText.eprint = lambda x: None


files = glob.glob("./data/cc-news-2020/01/*.wet.gz")
files.extend(glob.glob("./data/cc-news-2020/02/*.wet.gz"))
files.extend(glob.glob("./data/cc-news-2020/03/*.wet.gz"))
files.extend(glob.glob("./data/cc-news-2020/04/*.wet.gz"))

def whitelist(file):
    basename = os.path.basename(file)
    wh_dirname = os.path.join("./whitelists/", os.path.dirname(file).replace("data/", ""))
    if not os.path.exists(wh_dirname):
        os.makedirs(wh_dirname, exist_ok=True)
    whitelist_name = basename.split('.')[0]+"_en_whitelist.txt"
    whitelist_temp = basename.split('.')[0]+"_en_whitelist.tmp"
    # Real filename
    whitelist_name = os.path.join(wh_dirname, whitelist_name)
    # Temporary filename
    whitelist_temp = os.path.join(wh_dirname, whitelist_temp)
    if os.path.exists(whitelist_name):
        print(f"Skipping {file}, already done")
    else:
        before_inner = datetime.datetime.now()
        print(f"Computing whitelist for {file}")
        with open(whitelist_temp, "w") as w:
            with open(file, 'rb') as f:
                for record in ArchiveIterator(f):
                    if "WARC-Refers-To" in record.headers:
                        body = record.reader.read().decode()
                        lang = detect(body.replace("\n", " "))['lang']
                        if lang == "en":
                            w.write(record.headers['WARC-Refers-To']+"\n")
        after_inner = datetime.datetime.now()
        # Rename the temp file to the real filename once the computation is done
        os.rename(whitelist_temp, whitelist_name)
        print(f"Computed whitelist for {file} in {after_inner-before_inner}")


print("Computing english whitelists for TREC 2020 collection")
before = datetime.datetime.now()
with Pool(4) as pool:
    counts = pool.map(whitelist, files)
after = datetime.datetime.now()
print(f"Done in: {after-before}")


print("Joining the whitelist .txt files in a single file")
input_files = glob.glob("./whitelists/cc-news-2020/**/*.txt")
my_cmd = ['cat'] + input_files
with open('./data/cc-news_en_whitelist.txt', "w") as outfile:
    subprocess.run(my_cmd, stdout=outfile)
print("Done")
