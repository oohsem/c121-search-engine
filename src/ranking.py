"""Log-weighted vector-space ranking for the Developer corpus."""
from array import array
from collections import Counter
from heapq import heapify, heappop, nsmallest
from math import isfinite, sqrt
from operator import itemgetter

from hits import HITS_ROOT_SIZE
from positions import quoted_phrases, smallest_window
from search_index import IndexArtifactError
from term_weights import (
    document_weight,
    document_weights as document_weights,
    log_tf as log_tf,
    tf_idf,
)
from tokenizer import process_text


DEFAULT_IMPORTANT_BOOST = 0.35
DEFAULT_ANCHOR_WEIGHT = 0.12
DEFAULT_HITS_WEIGHT = 0.15
DEFAULT_PAGERANK_WEIGHT = 0.08
DEFAULT_PROXIMITY_WEIGHT = 0.10
DEFAULT_SHORT_PAGE_FLOOR = 85
PROXIMITY_SHORTLIST = 50
PHRASE_RANDOM_LOOKUPS = 128
PHRASE_DENSE_CANDIDATES = 10_000


STOPWORDS = frozenset(process_text(
    "a an and are as at be by for from in is it of on or that the to was were with"
))


def query_terms(query, filter_stopwords):
    terms = process_text(query)
    if not filter_stopwords or '"' in query:
        return Counter(terms)
    informative = [term for term in terms if term not in STOPWORDS]
    # Keep the whole query if removing stopwords would gut a short phrase
    # or leave no usable terms. All terms remain present in the index.
    if len(informative) >= 2 and 3 * (len(terms) - len(informative)) <= len(terms):
        return Counter(informative)
    return Counter(terms)


def rank_order(item, matches):
    """Prefer greater term coverage, then score, then numeric document ID."""
    doc, score = item
    return -matches.get(doc, 0), -score, doc


class PhraseIndexRequired(IndexArtifactError):
    """Exact phrase matching is unavailable until this index is rebuilt."""


class Ranker:
    """Rank matches with the course's lnc.ltc weighting."""

    def __init__(self, index, *, important_boost=DEFAULT_IMPORTANT_BOOST, filter_stopwords=True,
                 prefer_coverage=True, short_page_floor=DEFAULT_SHORT_PAGE_FLOOR,
                 pagerank_weight=DEFAULT_PAGERANK_WEIGHT,
                 proximity_weight=DEFAULT_PROXIMITY_WEIGHT, collapse_duplicates=True,
                 anchor_weight=DEFAULT_ANCHOR_WEIGHT, hits_weight=DEFAULT_HITS_WEIGHT):
        if not isfinite(important_boost) or important_boost < 0:
            raise ValueError("Important-text boost must be finite and nonnegative.")
        if type(short_page_floor) is not int or short_page_floor < 0:
            raise ValueError("The short-page threshold must be a nonnegative integer.")
        if not isfinite(pagerank_weight) or not 0.0 <= pagerank_weight <= 1.0:
            raise ValueError("PageRank weight must be between zero and one.")
        if not isfinite(proximity_weight) or not 0.0 <= proximity_weight <= 1.0:
            raise ValueError("Proximity weight must be between zero and one.")
        if not isfinite(anchor_weight) or not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("Anchor text weight must be between zero and one.")
        if not isfinite(hits_weight) or not 0.0 <= hits_weight <= 1.0:
            raise ValueError("HITS weight must be between zero and one.")
        self.index = index
        self.duplicate_groups = getattr(index, "duplicate_groups", None) if collapse_duplicates else None
        self.proximity_weight = proximity_weight
        self.anchor_weight = anchor_weight
        self.hits_weight = hits_weight if getattr(index, "link_graph", None) is not None else 0
        authority = index.pagerank_scores() if pagerank_weight else None
        self.authority_bonus = (
            array("d", (pagerank_weight * rank / (1.0 + rank) for rank in authority))
            if authority is not None else None
        )
        self.important_boost = important_boost
        self.filter_stopwords = filter_stopwords
        self.prefer_coverage = prefer_coverage
        self.content_weights = (
            array("d", (sqrt(min(1.0, length / short_page_floor))
                        for length in index.document_lengths()))
            if short_page_floor > 0 else None
        )
        self.norms = array("d")
        cross_weight = 2 * important_boost
        squared_weight = important_boost * important_boost
        for base_squared, cross, emphasis_squared in index.norm_components():
            squared_norm = base_squared + cross_weight * cross + squared_weight * emphasis_squared
            if not isfinite(squared_norm) or squared_norm < 0:
                raise IndexArtifactError("Invalid document norm or important-text boost.")
            self.norms.append(sqrt(squared_norm))

    def _plan_query(self, terms):
        collection_size = len(self.norms)
        plan = []
        for term, frequency in terms.items():
            location = self.index.posting_range(term)
            if location is None:
                continue
            weight = tf_idf(frequency, collection_size, location[1])
            # An IDF-zero term can still make a document a candidate.
            plan.append((term, location, weight))
        plan.sort(key=lambda item: (item[1][1], item[0]))
        return plan

    def _score_query(self, terms, eligible=None):
        plan = self._plan_query(terms)
        if not plan:
            return {}, {}
        query_norm = sqrt(sum(weight * weight for _, _, weight in plan))
        scores = {}
        matches = {} if self.prefer_coverage else None
        for term, location, weight in plan:
            normalized_weight = weight / query_norm if query_norm else 0.0
            for doc, frequency, important in self.index.iter_postings(term, location):
                if eligible is not None and doc not in eligible:
                    continue
                if doc >= len(self.norms) or not self.norms[doc]:
                    raise IndexArtifactError(f"Missing document norm for posting {doc}.")
                coordinate = document_weight(frequency, important, self.important_boost)
                scores[doc] = scores.get(doc, 0.0) + normalized_weight * coordinate
                if matches is not None:
                    matches[doc] = matches.get(doc, 0) + 1
        for doc in scores:
            scores[doc] /= self.norms[doc]
            if self.content_weights is not None:
                scores[doc] *= self.content_weights[doc]
            if self.authority_bonus is not None:
                scores[doc] += self.authority_bonus[doc]
        return scores, matches or {}

    def ranked_terms(self, terms, top_k=5):
        return self._rank(Counter(terms), top_k)

    def _add_anchor_scores(self, terms, scores):
        if not terms or not self.anchor_weight or getattr(self.index, "anchors", None) is None:
            return set()
        anchor_scores = {}
        scale = self.anchor_weight / len(terms)
        for term in terms:
            for doc, sources in self.index.anchor_postings(term):
                if doc >= len(self.norms):
                    raise IndexArtifactError(f"Unknown document in anchor posting {doc}.")
                anchor_scores[doc] = anchor_scores.get(doc, 0.0) + scale * sources / (1 + sources)
        anchor_only = anchor_scores.keys() - scores.keys()
        for doc, score in anchor_scores.items():
            scores[doc] = scores.get(doc, 0.0) + score
        return anchor_only

    def _rank(self, terms, top_k, phrases=()):
        if top_k <= 0:
            return []
        if phrases:
            return self._rank_phrases(terms, phrases, top_k)
        scores, matches = self._score_query(terms)
        anchor_only = self._add_anchor_scores(terms, scores)
        limit = max(top_k, HITS_ROOT_SIZE) if self.hits_weight else top_k

        def order(item):
            return (item[0] in anchor_only, *rank_order(item, matches))

        baseline = self._top_unique(scores.items(), order, limit)
        positions = self.index.positions
        term_count = sum(terms.values())
        if positions is None or not self.proximity_weight or term_count < 2:
            return self._finish_rank(baseline, order, top_k, anchor_only)
        locations = {term: self.index.posting_range(term) for term in terms}
        if None in locations.values():
            return self._finish_rank(baseline, order, top_k, anchor_only)

        # Include the unique baseline hits even if duplicates fill the shortlist.
        candidates = dict(nsmallest(max(limit, PROXIMITY_SHORTLIST), scores.items(), key=order))
        candidates.update(baseline)
        rescored = []
        for doc, score in candidates.items():
            rows = positions.rows(doc, locations, terms) if doc not in anchor_only else None
            if rows is not None:
                span = smallest_window(terms, lambda term: positions.values(rows[term]))
                if span is not None:
                    score += self.proximity_weight * term_count / span
            rescored.append((doc, score))
        ranked = self._top_unique(rescored, order, limit)
        return self._finish_rank(ranked, order, top_k, anchor_only)

    def _finish_rank(self, ranked, order, top_k, anchor_only=()):
        if not self.hits_weight or not ranked:
            return ranked[:top_k]
        graph_scores = self.index.hits_scores(doc for doc, _ in ranked if doc not in anchor_only)
        if not graph_scores:
            return ranked[:top_k]
        reranked = []
        for doc, score in ranked:
            hub, authority = graph_scores.get(doc, (0.0, 0.0))
            reranked.append((doc, score + self.hits_weight * (0.2 * hub + 0.8 * authority)))
        return nsmallest(top_k, reranked, key=order)

    def _result_keys(self, doc):
        url = self.index.document_url(str(doc))
        group = self.duplicate_groups[doc] if self.duplicate_groups is not None else doc
        return url, group

    def _top_unique(self, items, order, top_k, seen_urls=None, seen_groups=None):
        if self.duplicate_groups is None:
            return nsmallest(top_k, self._best_by_url(items, order).values(), key=order)

        seen_urls = set() if seen_urls is None else seen_urls
        seen_groups = set() if seen_groups is None else seen_groups
        candidates = [(order(item), item) for item in items]
        heapify(candidates)
        results = []
        while candidates and len(results) < top_k:
            _, item = heappop(candidates)
            url, group = self._result_keys(item[0])
            if url and url not in seen_urls and group not in seen_groups:
                seen_urls.add(url)
                seen_groups.add(group)
                results.append(item)
        return results

    def _best_by_url(self, items, order):
        best = {}
        for item in items:
            url = self.index.document_url(str(item[0]))
            if not url:
                continue
            previous = best.get(url)
            if previous is None or order(item) < order(previous):
                best[url] = item
        return best

    def _rank_phrases(self, terms, phrases, top_k):
        limit = max(top_k, HITS_ROOT_SIZE) if self.hits_weight else top_k
        positions = self.index.positions
        if positions is None:
            raise PhraseIndexRequired(
                "Exact phrase search needs token positions. Run python src/main.py to rebuild the index."
            )

        required = Counter()
        for phrase in phrases:
            required |= Counter(phrase)
        locations = {term: self.index.posting_range(term) for term in required}
        if None in locations.values():
            return []
        locations = dict(sorted(locations.items(), key=lambda item: item[1][1]))
        phrase_plans = [
            sorted(enumerate(phrase), key=lambda item: locations[item[1]][1]) for phrase in phrases
        ]
        anchor = next(iter(locations))
        eligible = {
            doc for doc, frequency, _ in self.index.iter_postings(anchor, locations[anchor])
            if frequency >= required[anchor]
        }
        for term, location in locations.items():
            if term != anchor:
                eligible.intersection_update(
                    doc for doc, frequency, _ in self.index.iter_postings(term, location)
                    if frequency >= required[term]
                )
            if not eligible:
                return []

        def phrase_match(rows):
            return rows is not None and all(
                positions.matches_phrase(plan, rows) for plan in phrase_plans
            )

        # A long exact phrase can reject most documents even when all its
        # words are common. Verify it before scoring a dense candidate set.
        verify_first = len(eligible) >= PHRASE_DENSE_CANDIDATES and any(len(plan) >= 4 for plan in phrase_plans)
        if verify_first:
            cursors = dict.fromkeys(locations, 0)
            eligible = {
                doc for doc in sorted(eligible)
                if phrase_match(positions.rows(doc, locations, required, cursors))
            }
            if not eligible:
                return []
        scores, matches = self._score_query(terms, eligible)

        def order(item):
            return rank_order(item, matches)

        if verify_first:
            return self._finish_rank(self._top_unique(scores.items(), order, limit), order, top_k)

        candidates = [order(item) for item in scores.items()]
        heapify(candidates)
        seen_urls = set()
        seen_groups = set()
        results = []

        for _ in range(PHRASE_RANDOM_LOOKUPS):
            if not candidates or len(results) == limit:
                return self._finish_rank(results, order, top_k)
            _, negative_score, doc = heappop(candidates)
            url, group = self._result_keys(doc)
            if not url or url in seen_urls or group in seen_groups:
                continue
            if phrase_match(positions.rows(doc, locations, required)):
                seen_urls.add(url)
                seen_groups.add(group)
                results.append((doc, -negative_score))

        if len(results) == limit:
            return self._finish_rank(results, order, top_k)

        # Rare phrases made of frequent words may need many candidates.
        # Scan the remaining postings in document order to reuse term cursors.
        cursors = dict.fromkeys(locations, 0)
        remaining = {}
        for item in sorted(candidates, key=itemgetter(2)):
            doc = item[2]
            url, group = self._result_keys(doc)
            if not url or url in seen_urls or group in seen_groups:
                continue
            key = (url, group) if self.duplicate_groups is not None else url
            previous = remaining.get(key)
            if (previous is None or item < previous) and phrase_match(
                positions.rows(doc, locations, required, cursors)
            ):
                remaining[key] = item
        if self.duplicate_groups is None:
            results.extend((doc, -score) for _, score, doc in nsmallest(limit - len(results), remaining.values()))
            return self._finish_rank(results, order, top_k)

        results.extend(self._top_unique(
            ((doc, -score) for _, score, doc in remaining.values()),
            order, limit - len(results), seen_urls, seen_groups,
        ))
        return self._finish_rank(results, order, top_k)

    def ranked_documents(self, query, top_k=5):
        return self._rank(query_terms(query, self.filter_stopwords), top_k, quoted_phrases(query))

    def search(self, query, top_k=5):
        return [self.index.document_url(str(doc)) for doc, _ in self.ranked_documents(query, top_k)]
