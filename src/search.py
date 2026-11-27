import argparse
from heapq import nsmallest
from pathlib import Path
from time import perf_counter

from ranking import Ranker
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


def main(argv=None, *, boolean=False):
    parser = argparse.ArgumentParser(description="Search the indexed Developer corpus.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--boolean", action="store_true", default=boolean,
                        help="Run the Milestone 2 Boolean search.")
    args = parser.parse_args(argv)

    try:
        with IndexReader(args.index_dir) as index:
            ranker = None if args.boolean else Ranker(index)
            while True:
                try:
                    raw_query = input("Give Query: ")
                except EOFError:
                    break
                started = perf_counter()
                query = raw_query.strip()
                if not query:
                    break
                results = search(query, index=index) if ranker is None else ranker.search(query)
                elapsed = (perf_counter() - started) * 1000
                for rank, url in enumerate(results, 1):
                    print(f"{rank}. {url}")
                if not results:
                    print("No results.")
                print(f"Query time: {elapsed:.1f} ms")
    except (IndexArtifactError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
