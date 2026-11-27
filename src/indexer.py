
import json
import heapq

from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from tokenizer import count_terms


# Spill when either bound is reached.
DOCS_PER_PARTIAL = 5000
POSTINGS_PER_PARTIAL = 100_000


def build_index(all_text, important_text, doc_num, index):

    # Count stems as they are read so a large document does not retain
    # a list of every token and a second list of every stem.
    term_frequencies = count_terms(all_text)
    important_frequencies = count_terms(important_text)

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


def merge_runs(partial_files, output_path, lexicon_file=None):
    with ExitStack() as stack:
        inputs = [stack.enter_context(path.open(encoding="utf-8")) for path in partial_files]
        output = stack.enter_context(output_path.open("wb"))
        heap = []
        term_count = 0
        if lexicon_file is not None:
            lexicon_file.write("{")

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

            if lexicon_file is not None:
                prefix = "," if term_count else ""
                key = json.dumps(term, ensure_ascii=False)
                lexicon_file.write(f"{prefix}{key}:{output.tell()}")
            record = {"term": term, "postings": merged_postings}
            output.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
            term_count += 1
        if lexicon_file is not None:
            lexicon_file.write("}")
        return term_count


def merge_partial_indexes(partial_files, final_path, lexicon_path):
    partial_files = [Path(path) for path in partial_files]
    final_path = Path(final_path)

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

        # Stream the legacy term/offset mapping as the final postings are written.
        with Path(lexicon_path).open("w", encoding="utf-8") as lexicon_file:
            return merge_runs(partial_files, final_path, lexicon_file)
