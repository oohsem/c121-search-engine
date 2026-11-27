import argparse
from heapq import nsmallest
from pathlib import Path

from search_index import DEFAULT_INDEX_DIR, IndexArtifactError, IndexReader
from tokenizer import process_text


def search(query, top_k=5, *, index=None):
    terms = list(dict.fromkeys(process_text(query)))
    if not terms:
        return []
    if index is None:
        with IndexReader() as reader:
            return search_terms(terms, top_k, reader)
    return search_terms(terms, top_k, index)


def search_terms(terms, top_k, index):
    all_postings = []
    for term in terms:
        postings = index.get_postings(term)
        if not postings:
            return []
        all_postings.append(postings)

    all_postings.sort(key=len)
    matches = set(all_postings[0])
    for postings in all_postings[1:]:
        matches.intersection_update(postings)
        if not matches:
            return []

    # The Boolean baseline has no relevance score. Use numeric document IDs
    # as a reproducible order before taking the requested number of results.
    return [index.document_url(doc_id) for doc_id in nsmallest(top_k, matches, key=int)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search indexed pages with Boolean AND.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)

    try:
        with IndexReader(args.index_dir) as index:
            while True:
                try:
                    query = input("Give Query: ").strip()
                except EOFError:
                    break
                if not query:
                    break
                for rank, url in enumerate(search(query, index=index)):
                    print(rank, ". ", url)
    except IndexArtifactError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
