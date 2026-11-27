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



def document_weights(frequency, important_frequency):
    """Split a document coordinate into its base and HTML emphasis weights.

    All occurrences contribute to the base. The second component is an
    additional boost for occurrences in the title, headings, or bold text.
    """
    return log_tf(frequency), log_tf(important_frequency)


def document_weight(frequency, important_frequency, important_boost):
    base, emphasis = document_weights(frequency, important_frequency)
    return base + important_boost * emphasis
