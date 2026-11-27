"""Log-weighted vector-space ranking for the Developer corpus."""
from array import array
from collections import Counter
from functools import lru_cache
from heapq import nsmallest
from math import isfinite, log10, sqrt

from search_index import IndexArtifactError
from tokenizer import process_text


DEFAULT_IMPORTANT_BOOST = 0.35
DEFAULT_SHORT_PAGE_FLOOR = 85


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


@lru_cache(maxsize=8192)
def log_tf(frequency):
    """A single occurrence weighs one, and an absent term weighs zero."""
    return 1.0 + log10(frequency) if frequency > 0 else 0.0


def tf_idf(frequency, documents, document_frequency):
    if not 0 < document_frequency <= documents:
        return 0.0
    return log_tf(frequency) * log10(documents / document_frequency)



def document_weights(frequency, important_frequency):
    """Split a document coordinate into its base and HTML emphasis weights.

    All occurrences contribute to the base. The second component is an
    additional boost for occurrences in the title, headings, or bold text.
    """
    return log_tf(frequency), log_tf(important_frequency)


def document_weight(frequency, important_frequency, important_boost):
    base, emphasis = document_weights(frequency, important_frequency)
    return base + important_boost * emphasis



class Ranker:
    """Rank matches with the course's lnc.ltc weighting."""

    def __init__(self, index, *, important_boost=DEFAULT_IMPORTANT_BOOST, filter_stopwords=True,
                 prefer_coverage=True, short_page_floor=DEFAULT_SHORT_PAGE_FLOOR):
        if not isfinite(important_boost) or important_boost < 0:
            raise ValueError("Important-text boost must be finite and nonnegative.")
        if type(short_page_floor) is not int or short_page_floor < 0:
            raise ValueError("The short-page threshold must be a nonnegative integer.")
        self.index = index
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

    def _plan_query(self, query):
        collection_size = len(self.norms)
        plan = []
        for term, frequency in query_terms(query, self.filter_stopwords).items():
            location = self.index.posting_range(term)
            if location is None:
                continue
            weight = tf_idf(frequency, collection_size, location[1])
            # An IDF-zero term can still make a document a candidate.
            plan.append((term, location, weight))
        plan.sort(key=lambda item: (item[1][1], item[0]))
        return plan

    def _score_query(self, query):
        plan = self._plan_query(query)
        if not plan:
            return {}, {}
        query_norm = sqrt(sum(weight * weight for _, _, weight in plan))
        scores = {}
        matches = {} if self.prefer_coverage else None
        for term, location, weight in plan:
            normalized_weight = weight / query_norm if query_norm else 0.0
            for doc, frequency, important in self.index.iter_postings(term, location):
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
        return scores, matches or {}

    def ranked_documents(self, query, top_k=5):
        scores, matches = self._score_query(query)
        # Rank full matches first and fill remaining slots with progressively
        # fewer terms. Cosine and numeric document ID order each tier.
        return nsmallest(
            top_k, scores.items(),
            key=lambda item: (-matches.get(item[0], 0), -item[1], item[0]),
        )

    def search(self, query, top_k=5):
        return [self.index.document_url(str(doc)) for doc, _ in self.ranked_documents(query, top_k)]
