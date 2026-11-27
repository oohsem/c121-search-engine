
from collections import Counter
from functools import lru_cache

from nltk.stem import PorterStemmer

# Create the stemmer once so we can reuse it.
stemmer = PorterStemmer()
stem_word = lru_cache(maxsize=32768)(stemmer.stem)


def iter_tokens(text):
    current_token = ""
    for character in text:
        if character.isalnum():
            current_token += character.lower()
        elif current_token:
            yield current_token
            current_token = ""
    if current_token:
        yield current_token


def tokenize(text):
    return list(iter_tokens(text))


def count_terms(text):
    return Counter(stem_word(token) for token in iter_tokens(text))


def stem_tokens(tokens):
    stemmed_tokens = []

    # Stem each token using the Porter stemmer.
    for token in tokens:
        stemmed_word = stem_word(token)
        stemmed_tokens.append(stemmed_word)

    return stemmed_tokens


def process_text(text):
    # Tokenize and stem the input text.
    tokens = tokenize(text)
    stemmed_tokens = stem_tokens(tokens)

    return stemmed_tokens
