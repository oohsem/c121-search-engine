import json
from pathlib import Path
from urllib.parse import urldefrag
from parser import parse_html
from indexer import DOCS_PER_PARTIAL, build_index, save_partial_index, merge_partial_indexes

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

FINAL_DIR = PROJECT_DIR / "final_index"

PARTIAL_DIR = PROJECT_DIR / "partial_indexes"

FINAL_INDEX_PATH = FINAL_DIR / "index.jsonl"

DOCUMENTS_PATH = FINAL_DIR / "documents.jsonl"

LEXICON_PATH = FINAL_DIR/ 'lexicon.json'

REPORT_PATH = FINAL_DIR / "milestone1_report.txt"


def get_json_files():

    # Find all JSON files, including files in subdirectories.
    return DATASET_DIR.rglob("*.json")


def read_document(file_path):

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
        doc_id = file_path.relative_to(DATASET_DIR).as_posix()

        return {
            "doc_id": doc_id,
            "url": url,
            "content": content
        }

    except (json.JSONDecodeError, OSError) as error:

        print("Error reading:", file_path, error)

        return None


def generate_report(document_count,unique_token_count,partial_count,skipped_count,index_size_kb):

    # Create the report contents.
    report = []

    report.append("MILESTONE 1: INVERTED INDEX REPORT")
    report.append("====================================")
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

    report.append(str(FINAL_INDEX_PATH))
    report.append(str(DOCUMENTS_PATH))
    report.append(str(LEXICON_PATH))

    # Write the report to a text file.
    with open(REPORT_PATH, "w", encoding="utf-8") as file:

        for line in report:
            file.write(line + "\n")

    print("\nReport generated at:", REPORT_PATH)


def main():

    # Check whether the dataset directory exists.
    if not DATASET_DIR.exists():

        print("Dataset directory not found:", DATASET_DIR)

        return

    # Set this to an integer to limit processing during testing.
    # Keep None to process the entire dataset.
    max_docs = None

    # Create the output directories.
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old partial indexes from previous runs.
    for old_file in PARTIAL_DIR.glob("partial_*.jsonl"):
        old_file.unlink()

    print("Starting index construction...")

    # In-memory inverted index.
    inverted_index = {}

    document_count = 0
    skipped_count = 0
    partial_count = 0

    # Store document IDs and URLs separately.
    with open(
        DOCUMENTS_PATH,
        "w",
        encoding="utf-8"
    ) as doc_file:

        for file_path in get_json_files():

            # Stop if the maximum document limit is reached.
            if max_docs is not None:
                if document_count >= max_docs:
                    break

            document = read_document(file_path)

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

                save_partial_index(inverted_index, PARTIAL_DIR, partial_count)

    # Save any remaining terms.
    if inverted_index:

        partial_count += 1

        print("Writing final partial index:", partial_count)

        save_partial_index(inverted_index, PARTIAL_DIR, partial_count)

    print("\nDocument processing complete.")

    print("Documents indexed:", document_count)

    print("Documents skipped:", skipped_count)

    print("Partial indexes created:", partial_count)

    # Check the assignment's minimum partial-index requirement.
    if partial_count < 3:

        print("WARNING: Fewer than three partial indexes were created.")

    # Find all generated partial indexes.
    partial_files = sorted(PARTIAL_DIR.glob("partial_*.jsonl"))

    # Merge the partial indexes.
    print("\nMerging partial indexes...")

    unique_token_count = merge_partial_indexes(partial_files, FINAL_INDEX_PATH, LEXICON_PATH)

    print("Merging complete.")

    # Calculate the total size of the final index files.
    index_size_bytes = (FINAL_INDEX_PATH.stat().st_size + DOCUMENTS_PATH.stat().st_size + LEXICON_PATH.stat().st_size)

    index_size_kb = index_size_bytes / 1024

    # Print the required analytics.
    print("\n========== INDEX ANALYTICS ==========")

    print("Number of indexed documents:", document_count)

    print("Number of unique tokens:", unique_token_count)

    print("Total index size (KB):", round(index_size_kb, 2))

    print("=====================================")

    # Generate the Milestone 1 report.
    generate_report(document_count, unique_token_count, partial_count, skipped_count,index_size_kb)

    print("\nFinal index saved at:", FINAL_INDEX_PATH)


if __name__ == "__main__":
    main()

