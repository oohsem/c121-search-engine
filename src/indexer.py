
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

def merge_partial_indexes(partial_files, final_path):

    merged_term_count = 0

    # Open all partial index files.
    with ExitStack() as stack:

        files = []

        for partial_path in partial_files:
            file = stack.enter_context(
                open(partial_path, "r", encoding="utf-8")
            )
            files.append(file)

        # The heap keeps the next term from each file.
        heap = []

        # Read the first record from each partial index.
        for file_number, file in enumerate(files):

            line = file.readline()

            if line:
                record = json.loads(line)

                heapq.heappush(
                    heap,
                    (record["term"], file_number, record["postings"])
                )

        # Write the merged index.
        with open(final_path, "w", encoding="utf-8") as output:

            while heap:

                # Get the alphabetically smallest term.
                current_term = heap[0][0]

                merged_postings = {}

                # Collect postings for the same term.
                while heap and heap[0][0] == current_term:

                    term, file_number, postings = heapq.heappop(heap)

                    # Add this partial index's postings.
                    merged_postings.update(postings)

                    # Read the next record from that file.
                    line = files[file_number].readline()

                    if line:
                        record = json.loads(line)

                        heapq.heappush(
                            heap,
                            (
                                record["term"],
                                file_number,
                                record["postings"]
                            )
                        )

                # Write the merged term and its postings.
                record = {
                    "term": current_term,
                    "postings": merged_postings
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                merged_term_count += 1

    return merged_term_count
