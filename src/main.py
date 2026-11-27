import argparse
import json
from pathlib import Path
from urllib.parse import urldefrag
from parser import parse_html
from optimize import optimize
from indexer import DOCS_PER_PARTIAL, build_index, save_partial_index, merge_partial_indexes

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the CS121 inverted index.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--max-docs", type=positive_int, help="Limit a separate smoke build.")
    return parser.parse_args(argv)


def read_document(file_path, dataset_dir):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:

            data = json.load(file)

        # Make sure the JSON contains a dictionary.
        if not isinstance(data, dict):
            return None

        url = data.get("url", "")
        content = data.get("content", "")

        # Make sure URL and content are strings.
        if not isinstance(url, str):
            url = ""

        if not isinstance(content, str):
            content = ""

        # Remove URL fragments.
        url = urldefrag(url).url

        # Use the relative file path as the document ID.
        doc_id = file_path.relative_to(dataset_dir).as_posix()

        return {
            "doc_id": doc_id,
            "url": url,
            "content": content
        }

    except (json.JSONDecodeError, OSError) as error:

        print("Error reading:", file_path, error)

        return None


def generate_report(document_count, unique_token_count, partial_count, skipped_count,
                    index_size_kb, dataset_dir, max_docs, index_paths):

    # Create the report contents.
    report = []

    report.append("MILESTONE 1: INVERTED INDEX REPORT")
    report.append("====================================")
    report.append("")

    build_mode = "smoke" if max_docs is not None else "full"
    report.append(f"Build mode: {build_mode}")
    report.append(f"Dataset: {dataset_dir}")
    report.append(f"Document limit: {max_docs if max_docs is not None else 'none'}")
    report.append("")

    report.append("INDEX ANALYTICS")
    report.append("----------------")

    report.append(
        "Number of indexed documents: " + str(document_count)
    )

    report.append(
        "Number of unique tokens: " + str(unique_token_count)
    )

    report.append(
        "Total index size (KB): " + str(round(index_size_kb, 2))
    )

    report.append("")

    report.append("ADDITIONAL INFORMATION")
    report.append("----------------------")

    report.append(
        "Number of partial indexes created: " + str(partial_count)
    )

    report.append(
        "Number of documents skipped: " + str(skipped_count)
    )

    report.append("")

    report.append("INDEX FILES")
    report.append("-----------")

    report.extend(str(path) for path in index_paths)
    report_path = index_paths[0].parent / "milestone1_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nReport generated at:", report_path)


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    max_docs = args.max_docs
    output_root = args.output_root.resolve()
    if max_docs is not None:
        output_root /= "smoke_index"
    final_dir = output_root / "final_index"
    partial_dir = output_root / "partial_indexes"
    index_path = final_dir / "index.jsonl"
    documents_path = final_dir / "documents.jsonl"
    lexicon_path = final_dir / "lexicon.json"
    final_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    print("Build mode:", "smoke" if max_docs is not None else "full")
    print("Dataset:", dataset_dir)
    print("Document limit:", max_docs if max_docs is not None else "none")

    # Remove old partial indexes from previous runs.
    for old_file in partial_dir.glob("partial_*.jsonl"):
        old_file.unlink()

    print("Starting index construction...")

    # In-memory inverted index.
    inverted_index = {}

    document_count = 0
    skipped_count = 0
    partial_count = 0

    # Store document IDs and URLs separately.
    with open(
        documents_path,
        "w",
        encoding="utf-8"
    ) as doc_file:

        for file_path in dataset_dir.rglob("*.json"):

            # Stop if the maximum document limit is reached.
            if max_docs is not None:
                if document_count >= max_docs:
                    break

            document = read_document(file_path, dataset_dir)

            # Skip documents that could not be read.
            if document is None:

                skipped_count += 1

                continue

            doc_id = document["doc_id"]

            url = document["url"]

            html = document["content"]

            # Extract regular and important text.
            all_text, important_text = parse_html(html)

            # Add document terms to the inverted index.
            build_index(all_text, important_text, document_count, inverted_index)

            # Save document metadata.
            doc_record = {'id': document_count, "doc_id": doc_id, "url": url}

            doc_file.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            document_count += 1

            # Display progress every 1,000 documents.
            if document_count % 1000 == 0:

                print("Processed documents:", document_count)

            # Save a partial index every 5,000 documents.
            if document_count % DOCS_PER_PARTIAL == 0:

                partial_count += 1

                print("Writing partial index:", partial_count)

                save_partial_index(inverted_index, partial_dir, partial_count)

    # Save any remaining terms.
    if inverted_index:

        partial_count += 1

        print("Writing final partial index:", partial_count)

        save_partial_index(inverted_index, partial_dir, partial_count)

    print("\nDocument processing complete.")

    print("Documents indexed:", document_count)

    print("Documents skipped:", skipped_count)

    print("Partial indexes created:", partial_count)

    # Check the assignment's minimum partial-index requirement.
    if max_docs is None and partial_count < 3:

        print("WARNING: Fewer than three partial indexes were created.")

    # Find all generated partial indexes.
    partial_files = sorted(partial_dir.glob("partial_*.jsonl"))

    # Merge the partial indexes.
    print("\nMerging partial indexes...")

    unique_token_count = merge_partial_indexes(partial_files, index_path, lexicon_path)

    print("Merging complete.")

    # Calculate the total size of the final index files.
    index_size_bytes = (index_path.stat().st_size + documents_path.stat().st_size + lexicon_path.stat().st_size)

    index_size_kb = index_size_bytes / 1024

    # Print the required analytics.
    print("\n========== INDEX ANALYTICS ==========")

    print("Number of indexed documents:", document_count)

    print("Number of unique tokens:", unique_token_count)

    print("Total index size (KB):", round(index_size_kb, 2))

    print("=====================================")

    # Generate the Milestone 1 report.
    generate_report(document_count, unique_token_count, partial_count, skipped_count,
                    index_size_kb, dataset_dir, max_docs, (index_path, documents_path, lexicon_path))

    print("\nFinal index saved at:", index_path)

    # Build the compact disk-backed index required by Milestone 3.
    optimize(final_dir)


if __name__ == "__main__":
    main()

