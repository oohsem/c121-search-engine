import argparse
import math
from heapq import nlargest
from pathlib import Path

from search_index import (DEFAULT_INDEX_DIR,IndexArtifactError,CompactIndexReader)
from tokenizer import process_text


def search(query, top_k=5, *, index=None):
    terms = list(dict.fromkeys(process_text(query)))
    if not terms:
        return []
    if index is None:
        with CompactIndexReader() as reader:
            return search_terms(terms, top_k, reader)
    return search_terms(terms, top_k, index)


def calculate_tfidf(tf, important_tf, df, total_docs):

    tf_weight = 1 + math.log(tf)

    idf_weight = math.log(
        (total_docs + 1) / (df + 1)
    ) + 1

    # Boost words appearing in important HTML elements.
    important_weight = 1.5 * math.log1p(important_tf)

    return (tf_weight + important_weight) * idf_weight


def search_terms(terms, top_k, index):

    if not terms or top_k <= 0:
        return []

    all_postings = []

    # Retrieve postings for each query term.
    for term in terms:

        postings = index.get_postings(term)

        if not postings:
            return []

        all_postings.append(postings)

    # Start with the smallest posting list.
    all_postings.sort(key=len)

    matches = set(all_postings[0])

    # Find documents containing every query term.
    for postings in all_postings[1:]:

        matches.intersection_update(postings)

        if not matches:
            return []

    total_docs = len(index.doc_urls)

    scores = {}

    # Calculate the relevance score of each matching document.
    for doc_id in matches:

        score = 0.0

        for postings in all_postings:

            tf = postings[doc_id]["tf"]
            important_tf = postings[doc_id]["important_tf"]
            df = len(postings)

            score += calculate_tfidf(
                tf, important_tf, df, total_docs
)

        # Normalize scores based on document length.
        doc_length = index.lengths[int(doc_id)]

        normalization = (
            0.75 + 0.25 * math.sqrt(
                doc_length / index.avg_length
            )
        )

        scores[doc_id] = score / normalization


    # Keep the highest-scoring document for each unique URL.

    best_by_url = {}

    for doc_id in matches:


        url = index.document_url(doc_id)

        if not url:
            continue

        previous_doc = best_by_url.get(url)

        if previous_doc is None or (
            scores[doc_id], -int(doc_id)
        ) > (
            scores[previous_doc], -int(previous_doc)
        ):
            best_by_url[url] = doc_id

    # Select the highest-scoring unique documents.
    ranked_docs = nlargest(
        top_k,
        best_by_url.values(),
        key=lambda doc_id: (
            scores[doc_id],
            -int(doc_id)
        )
    )

    return [
        index.document_url(doc_id)
        for doc_id in ranked_docs
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search indexed pages with Boolean AND.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)

    try:
        with CompactIndexReader(args.index_dir) as index:
            while True:
                try:
                    query = input("Give Query: ").strip()
                except EOFError:
                    break
                if not query:
                    break
                for rank, url in enumerate(
                    search(query, index=index),
                    start=1
                ):
                    print(f"{rank}. {url}")
    except IndexArtifactError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
