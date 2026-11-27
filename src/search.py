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
    if index.compact is not None:
        plan = []
        for term in terms:
            location = index.posting_range(term)
            if location is None:
                return []
            plan.append((term, location))
        plan.sort(key=lambda item: item[1][1])
        document_lists = (
            (doc_id for doc_id, _, _ in index.iter_postings(term, location))
            for term, location in plan
        )
    else:
        all_postings = [index.get_postings(term) for term in terms]
        if any(not postings for postings in all_postings):
            return []
        all_postings.sort(key=len)
        document_lists = (map(int, postings) for postings in all_postings)

    matches = set(next(document_lists))
    for documents in document_lists:
        matches.intersection_update(documents)
        if not matches:
            return []

    # Boolean search has no relevance score. Sort by numeric document ID.
    return [index.document_url(str(doc_id)) for doc_id in nsmallest(top_k, matches)]


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
