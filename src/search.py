import argparse
from heapq import nsmallest
from pathlib import Path
from time import perf_counter

from ranking import DEFAULT_IMPORTANT_BOOST, DEFAULT_SHORT_PAGE_FLOOR, Ranker
from search_index import DEFAULT_INDEX_DIR, IndexArtifactError, IndexReader
from snippets import matching_excerpt
from tokenizer import process_text


QUERY_LIMIT_MS = 300.0


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
    parser.add_argument("--keep-stopwords", action="store_true",
                        help="Keep every word from the ranked query.")
    parser.add_argument("--important-boost", type=float, default=DEFAULT_IMPORTANT_BOOST,
                        help="Weight given to words in titles, headings, and bold text.")
    parser.add_argument("--short-page-floor", type=int, default=DEFAULT_SHORT_PAGE_FLOOR,
                        help="Token threshold for the short-page penalty; 0 disables it.")
    parser.add_argument("--snippets", action="store_true",
                        help="Show short excerpts from local indexed pages in ranked mode.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_INDEX_DIR.parent / "dataset",
                        help="Folder containing the original pages for --snippets.")
    args = parser.parse_args(argv)
    dataset = args.dataset_dir.resolve() if args.snippets else None
    if dataset is not None and (args.boolean or not dataset.is_dir()):
        parser.error("--snippets needs ranked mode and an existing --dataset-dir.")

    try:
        with IndexReader(args.index_dir, include_sources=args.snippets) as index:
            ranker = None if args.boolean else Ranker(
                index, important_boost=args.important_boost,
                filter_stopwords=not args.keep_stopwords, short_page_floor=args.short_page_floor,
            )
            while True:
                try:
                    raw_query = input("Give Query: ")
                except EOFError:
                    break
                started = perf_counter()
                query = raw_query.strip()
                if not query:
                    break
                if ranker is None:
                    results = [(url, "") for url in search(query, index=index)]
                elif dataset is None:
                    results = [(url, "") for url in ranker.search(query)]
                else:
                    terms = set(process_text(query))
                    results = []
                    for doc, _ in ranker.ranked_documents(query):
                        url = index.document_url(str(doc))
                        excerpt = matching_excerpt(terms, dataset, index.doc_sources.get(doc), url)
                        results.append((url, excerpt))
                elapsed = (perf_counter() - started) * 1000
                for rank, (url, excerpt) in enumerate(results, 1):
                    print(f"{rank}. {url}")
                    if excerpt:
                        print(f"   {excerpt}")
                if not results:
                    print("No results.")
                limit_status = " (over 300 ms)" if elapsed > QUERY_LIMIT_MS else ""
                print(f"Query time: {elapsed:.1f} ms{limit_status}")
    except (IndexArtifactError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
