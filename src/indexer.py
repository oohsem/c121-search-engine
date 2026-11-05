
import json
import heapq

from pathlib import Path
from collections import Counter
from contextlib import ExitStack

from tokenizer import process_text


# Write a partial index after every 5,000 documents.
DOCS_PER_PARTIAL = 5000


def build_index(all_text, important_text, doc_id, index):

    # Process all text in the document.
    tokens = process_text(all_text)

    # Process only important text.
    important_tokens = process_text(important_text)

    # Count how many times each term appears.
    term_frequencies = Counter(tokens)

    # Count how many times each term appears in important text.
    important_frequencies = Counter(important_tokens)

    # Add each term to the inverted index.
    for term, frequency in term_frequencies.items():

        # Create a dictionary for the term if it does not exist.
        if term not in index:
            index[term] = {}

        # Store the document ID and term frequencies.
        index[term][doc_id] = {
            "tf": frequency,
            "important_tf": important_frequencies.get(term, 0)
        }


def save_partial_index(index, partial_dir, partial_number):

    # Create a filename for this partial index.
    partial_path = partial_dir / f"partial_{partial_number:05d}.jsonl"

    # Write every term and its postings to disk.
    with open(partial_path, "w", encoding="utf-8") as file:

        # Sort terms alphabetically.
        for term in sorted(index):

            record = {
                "term": term,
                "postings": index[term]
            }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Clear the in-memory index after saving it.
    index.clear()

    return partial_path

