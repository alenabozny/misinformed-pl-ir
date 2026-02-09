#!/usr/bin/env bash

INDEX_NAME="trec2020"
if [[ ! -d ./indices/$INDEX_NAME || -z "$(ls -A ./indices/$INDEX_NAME/)" ]]; then
    echo "Indexing $INDEX_NAME. Check the log file: ./logs/"$INDEX_NAME"_index.logs"
    python3 -m pyserini.index.lucene \
        --collection CommonCrawlWetCollection \
        --input ./data/cc-news-2020/ \
        --index indices/$INDEX_NAME \
        --generator WarcGenerator \
        --whitelist ./data/cc-news_en_whitelist.txt \
        --threads 12 > ./logs/"$INDEX_NAME"_index.logs 2>&1
else
   echo "Skipping index './indices/$INDEX_NAME/': already exist. Delete directory to re-index."
fi


INDEX_NAME="c4_anserini"
if [[ ! -d ./indices/$INDEX_NAME || -z "$(ls -A ./indices/$INDEX_NAME/)" ]]; then
    echo "Indexing $INDEX_NAME. Check the log file: ./logs/"$INDEX_NAME"_index.logs"
    python3 -m pyserini.index.lucene \
        --collection C4NoCleanCollection \
        --input ./data/c4/en.noclean/ \
        --index indices/$INDEX_NAME \
        --generator C4Generator \
        --threads 16 > ./logs/"$INDEX_NAME"_index.logs 2>&1
else
   echo "Skipping index './indices/$INDEX_NAME/': already exist. Delete directory to re-index."
fi
