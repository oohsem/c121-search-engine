
from nltk.stem import PorterStemmer

# Create the stemmer once so we can reuse it.
stemmer = PorterStemmer()


def tokenize(text):
    tokens = []
    current_token = ""

    for character in text:
        # Keep letters and numbers together as tokens.
        if character.isalnum():
            current_token += character.lower()

        else:
            # Save the token whenever we encounter a separator.
            if current_token:
                tokens.append(current_token)
                current_token = ""

    # Save the last token if the text ends with an alphanumeric character.
    if current_token:
        tokens.append(current_token)

    return tokens


def stem_tokens(tokens):
    stemmed_tokens = []

    # Stem each token using the Porter stemmer.
    for token in tokens:
        stemmed_word = stemmer.stem(token)
        stemmed_tokens.append(stemmed_word)

    return stemmed_tokens


def process_text(text):
    # Tokenize and stem the input text.
    tokens = tokenize(text)
    stemmed_tokens = stem_tokens(tokens)

    return stemmed_tokens
