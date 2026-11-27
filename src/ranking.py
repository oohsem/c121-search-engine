"""Log-weighted vector-space ranking for the Developer corpus."""
from functools import lru_cache
from math import log10


@lru_cache(maxsize=8192)
def log_tf(frequency):
    """A single occurrence weighs one, and an absent term weighs zero."""
    return 1.0 + log10(frequency) if frequency > 0 else 0.0


def tf_idf(frequency, documents, document_frequency):
    if not 0 < document_frequency <= documents:
        return 0.0
    return log_tf(frequency) * log10(documents / document_frequency)
