#!/usr/bin/env bash

retrieve_bm25 () {
    python -m pyserini.search.lucene --topics $1 \
     --index $3 \
     --output $2 \
     --hits 1000 \
     --bm25 --k1 0.9 --b 0.4
}

trec2020_bm25 () {
    printf "\nRetrieving documents for TREC 2020 HM with BM25\n"
    INDEX="./indices/trec2020"

    echo "Computing BM25 hits for 'title' queries"
    QUERIES="./queries/misinfo-2020-topics-title.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2020/run.misinfo-2020-title.bm25_en.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX
    sed -i -e 's/<urn:uuid://g' -e 's/>//g' $OUT_FILE

    echo "Computing BM25 hits for 'description' queries"
    QUERIES="./queries/misinfo-2020-topics-description.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2020/run.misinfo-2020-description.bm25_en.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX
    sed -i -e 's/<urn:uuid://g' -e 's/>//g' $OUT_FILE
}


trec2021_bm25 () {
    printf "\nRetrieving documents for TREC 2021 HM with BM25\n"
    INDEX="./indices/c4_anserini"

    echo "Computing BM25 hits for 'query' queries"
    QUERIES="./queries/misinfo-2021-topics-query.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2021/run.misinfo-2021-query.bm25.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX

    echo "Computing BM25 hits for 'description' queries"
    QUERIES="./queries/misinfo-2021-topics-description.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2021/run.misinfo-2021-description.bm25.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX
}

trec2022_bm25 () {
    printf "\nRetrieving documents for TREC 2022 HM with BM25\n"
    INDEX="./indices/c4_anserini"

    echo "Computing BM25 hits for 'query' queries"
    QUERIES="./queries/misinfo-2022-topics-query.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2022/run.misinfo-2022-query.bm25.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX

    echo "Computing BM25 hits for 'question' queries"
    QUERIES="./queries/misinfo-2022-topics-question.tsv"
    OUT_FILE="./misinfo-runs/adhoc/2022/run.misinfo-2022-question.bm25.txt"
    retrieve_bm25 $QUERIES $OUT_FILE $INDEX
}

trec2020_bm25
trec2021_bm25
trec2022_bm25
