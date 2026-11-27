import argparse
from heapq import nsmallest
from pathlib import Path
from sys import stderr
from time import perf_counter

from ranking import (
    DEFAULT_ANCHOR_WEIGHT,
    DEFAULT_IMPORTANT_BOOST,
    DEFAULT_PAGERANK_WEIGHT,
    DEFAULT_PROXIMITY_WEIGHT,
    DEFAULT_SHORT_PAGE_FLOOR,
    PhraseIndexRequired,
    Ranker,
)
from search_index import DEFAULT_INDEX_DIR, IndexArtifactError, IndexReader
from snippets import matching_excerpt
from term_weights import document_weight, tf_idf
from tokenizer import process_text


QUERY_LIMIT_MS = 300.0


def search(query, top_k=5, *, index=None):
    if not query.strip() or top_k <= 0:
        return []
    if index is None:
        with IndexReader() as reader:
            return Ranker(reader).search(query, top_k)
    return Ranker(index).search(query, top_k)


def search_terms(terms, top_k, index):
    if not terms or top_k <= 0:
        return []
    ranker = Ranker(index)
    return [index.document_url(str(doc)) for doc, _ in ranker.ranked_terms(terms, top_k)]


def calculate_tfidf(tf, important_tf, df, total_docs):
    return document_weight(tf, important_tf, DEFAULT_IMPORTANT_BOOST) * tf_idf(1, total_docs, df)


def boolean_search(query, top_k=5, *, index=None):
    terms = list(dict.fromkeys(process_text(query)))
    if not terms:
        return []
    if index is None:
        with IndexReader() as reader:
            return boolean_search_terms(terms, top_k, reader)
    return boolean_search_terms(terms, top_k, index)


def boolean_search_terms(terms, top_k, index):
    if not terms or top_k <= 0:
        return []
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
    parser = argparse.ArgumentParser(
        description='Search the indexed Developer corpus. Use double quotes for an exact phrase.',
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--boolean", action="store_true", default=boolean,
                        help="Run the Milestone 2 Boolean search.")
    parser.add_argument("--keep-stopwords", action="store_true",
                        help="Keep every word from the ranked query.")
    parser.add_argument("--important-boost", type=float, default=DEFAULT_IMPORTANT_BOOST,
                        help="Weight given to words in titles, headings, and bold text.")
    parser.add_argument("--anchor-weight", type=float, default=DEFAULT_ANCHOR_WEIGHT,
                        help="Weight of incoming anchor text; 0 disables it.")
    parser.add_argument("--pagerank-weight", type=float, default=DEFAULT_PAGERANK_WEIGHT,
                        help="Weight of offline link authority; 0 disables PageRank.")
    parser.add_argument("--proximity-weight", type=float, default=DEFAULT_PROXIMITY_WEIGHT,
                        help="Proximity boost for unquoted ranked queries; 0 disables it.")
    parser.add_argument("--show-duplicates", action="store_true",
                        help="Show separate URLs with matching content in ranked search.")
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
                pagerank_weight=args.pagerank_weight, proximity_weight=args.proximity_weight,
                collapse_duplicates=not args.show_duplicates,
                anchor_weight=args.anchor_weight,
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
                    results = [(url, "") for url in boolean_search(query, index=index)]
                else:
                    try:
                        hits = ranker.ranked_documents(query)
                    except PhraseIndexRequired as error:
                        print(error, file=stderr)
                        continue
                    terms = set(process_text(query)) if dataset is not None else None
                    results = []
                    for doc, _ in hits:
                        url = index.document_url(str(doc))
                        excerpt = (matching_excerpt(terms, dataset, index.doc_sources.get(doc), url)
                                   if dataset is not None else "")
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
