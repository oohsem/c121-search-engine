
import json
import heapq

from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from tokenizer import process_text


# Spill when either bound is reached.
DOCS_PER_PARTIAL = 5000
POSTINGS_PER_PARTIAL = 500_000


def build_index(all_text, important_text, doc_num, index):

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
        index[term][doc_num] = {
            "tf": frequency,
            "important_tf": important_frequencies.get(term, 0)
        }


    return len(term_frequencies)


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

MAX_MERGE_INPUTS = 16


def merge_runs(partial_files, output_path, lexicon=None):
    with ExitStack() as stack:
        inputs = [stack.enter_context(path.open(encoding="utf-8")) for path in partial_files]
        output = stack.enter_context(output_path.open("wb"))
        heap = []

        def read_next(file_number):
            line = inputs[file_number].readline()
            if line:
                record = json.loads(line)
                heapq.heappush(heap, (record["term"], file_number, record["postings"]))

        for file_number in range(len(inputs)):
            read_next(file_number)

        while heap:
            term = heap[0][0]
            merged_postings = {}
            while heap and heap[0][0] == term:
                _, file_number, postings = heapq.heappop(heap)
                merged_postings.update(postings)
                read_next(file_number)

            if lexicon is not None:
                lexicon[term] = output.tell()
            record = {"term": term, "postings": merged_postings}
            output.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))


def merge_partial_indexes(partial_files, final_path, lexicon_path):
    partial_files = [Path(path) for path in partial_files]
    final_path = Path(final_path)
    lexicon = {}

    # Limit the fan-in even when a large corpus creates many runs.
    with TemporaryDirectory(prefix="merge-", dir=final_path.parent) as directory:
        temp_dir = Path(directory)
        pass_number = 0
        while len(partial_files) > MAX_MERGE_INPUTS:
            next_runs = []
            for start in range(0, len(partial_files), MAX_MERGE_INPUTS):
                batch = partial_files[start:start + MAX_MERGE_INPUTS]
                if len(batch) == 1:
                    next_runs.append(batch[0])
                    continue
                merged_path = temp_dir / f"run-{pass_number}-{start}.jsonl"
                merge_runs(batch, merged_path)
                next_runs.append(merged_path)
                for path in batch:
                    if path.parent == temp_dir:
                        path.unlink()
            partial_files = next_runs
            pass_number += 1

        merge_runs(partial_files, final_path, lexicon)

    with Path(lexicon_path).open("w", encoding="utf-8") as lex_file:
        json.dump(lexicon, lex_file, ensure_ascii=False)
    return len(lexicon)
