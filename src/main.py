import argparse
import gc
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urldefrag, urlsplit
from uuid import uuid4

from index_storage import publish_generation
from parser import parse_html
from pagerank import PageLinks, write_pagerank
from optimize import optimize
from indexer import (
    DOCS_PER_PARTIAL,
    POSTINGS_PER_PARTIAL,
    build_index,
    merge_partial_indexes,
    save_partial_index,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"
LARGE_DOCUMENT_BYTES = 8 * 1024 * 1024

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
    parser.add_argument(
        "--max-postings", type=positive_int, default=POSTINGS_PER_PARTIAL,
        help="Spill after at least this many postings at a document boundary.",
    )
    return parser.parse_args(argv)


SKIP_REASONS = ("unreadable_file", "malformed_json", "unusable_url", "empty_content")


def read_document(file_path, dataset_dir):
    try:
        with file_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except OSError:
        return None, "unreadable_file"
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "malformed_json"

    if not isinstance(data, dict):
        return None, "malformed_json"

    raw_url = data.get("url")
    if not isinstance(raw_url, str):
        return None, "unusable_url"
    raw_url = raw_url.strip()
    address = raw_url.split("#", 1)[0]
    # urllib strips some controls while parsing. Check what we will store first.
    if any(ord(character) < 32 or ord(character) == 127 for character in address):
        return None, "unusable_url"
    try:
        url = urldefrag(raw_url).url
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        if parsed_url.scheme not in ("http", "https") or not hostname:
            return None, "unusable_url"
        if any(character.isspace() for character in hostname) or parsed_url.port == 0:
            return None, "unusable_url"
        # Reject ill-formed URLs before writing this generation's document map.
        url.encode("utf-8")
    except ValueError:
        return None, "unusable_url"

    content = data.get("content")
    if not isinstance(content, str) or not content or content.isspace():
        return None, "empty_content"

    return {
        "doc_id": file_path.relative_to(dataset_dir).as_posix(),
        "url": url,
        "content": content,
    }, None


def generate_report(document_count, unique_token_count, partial_count, skipped_by_reason,
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

    skipped_count = sum(skipped_by_reason.values())
    report.append(f"Number of JSON records examined: {document_count + skipped_count}")
    report.append(f"Number of documents skipped: {skipped_count}")
    for reason in SKIP_REASONS:
        report.append(f"  {reason}: {skipped_by_reason[reason]}")

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

    output_root = args.output_root.resolve()
    if args.max_docs is not None:
        output_root /= "smoke_index"
    final_root = output_root / "final_index"
    partial_root = output_root / "partial_indexes"
    final_dir = final_root / "generations" / f"index-{uuid4().hex}"
    final_dir.mkdir(parents=True)
    try:
        partial_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="build-", dir=partial_root) as directory:
            build_generation(args, dataset_dir, final_dir, Path(directory))
    except BaseException:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise

    # Readers keep their original generation even if another rebuild finishes.
    publish_generation(final_root, final_dir)
    print("\nFinal index saved at:", final_dir / "index.jsonl")


def spill_index(index, partial_dir, count, postings):
    count += 1
    print(f"Writing partial index {count}: {postings} postings")
    save_partial_index(index, partial_dir, count)
    return count


def build_generation(args, dataset_dir, final_dir, partial_dir):
    max_docs = args.max_docs
    index_path = final_dir / "index.jsonl"
    documents_path = final_dir / "documents.jsonl"
    lexicon_path = final_dir / "lexicon.json"
    links_path = partial_dir / "links.jsonl"

    print("Build mode:", "smoke" if max_docs is not None else "full")
    print("Dataset:", dataset_dir)
    print("Document limit:", max_docs if max_docs is not None else "none")

    # In-memory inverted index.
    inverted_index = {}

    document_count = 0
    skipped_by_reason = Counter()
    partial_count = 0
    docs_in_partial = 0
    postings_in_partial = 0

    # Store document IDs and URLs separately.
    with open(
        documents_path,
        "w",
        encoding="utf-8"
    ) as doc_file, links_path.open("w", encoding="utf-8", newline="\n") as link_file:

        for file_path in dataset_dir.rglob("*.json"):

            # Stop if the maximum document limit is reached.
            if max_docs is not None:
                if document_count >= max_docs:
                    break

            try:
                oversized = file_path.stat().st_size >= LARGE_DOCUMENT_BYTES
            except OSError:
                skipped_by_reason["unreadable_file"] += 1
                continue
            if oversized:
                if inverted_index:
                    partial_count = spill_index(
                        inverted_index, partial_dir, partial_count, postings_in_partial)
                    docs_in_partial = postings_in_partial = 0
                # Release old parser trees before decoding an unusually large page.
                gc.collect()

            document, reason = read_document(file_path, dataset_dir)
            if reason is not None:
                skipped_by_reason[reason] += 1
                continue

            doc_id = document["doc_id"]
            url = document["url"]
            links = PageLinks(url)
            all_text, important_text = parse_html(document.pop("content"), on_link=links.add)
            if not all_text or all_text.isspace():
                skipped_by_reason["empty_content"] += 1
                continue

            # Add document terms to the inverted index.
            postings_in_partial += build_index(all_text, important_text, document_count, inverted_index)
            del all_text, important_text
            docs_in_partial += 1

            # Save document metadata.
            doc_record = {'id': document_count, "doc_id": doc_id, "url": url}

            doc_file.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
            link_file.write(json.dumps(links.targets(), ensure_ascii=False) + "\n")
            del links

            document_count += 1

            # Display progress every 1,000 documents.
            if document_count % 1000 == 0:

                print("Processed documents:", document_count)

            # Complete a document before spilling so its postings stay together.
            if inverted_index and (
                docs_in_partial >= DOCS_PER_PARTIAL or postings_in_partial >= args.max_postings
            ):
                partial_count = spill_index(
                    inverted_index, partial_dir, partial_count, postings_in_partial)
                docs_in_partial = postings_in_partial = 0

    if inverted_index:
        partial_count = spill_index(inverted_index, partial_dir, partial_count, postings_in_partial)

    print("\nDocument processing complete.")

    print("Documents indexed:", document_count)

    skipped_count = sum(skipped_by_reason.values())
    print("JSON records examined:", document_count + skipped_count)
    print("Documents skipped:", skipped_count)
    for reason in SKIP_REASONS:
        print(f"  {reason}: {skipped_by_reason[reason]}")

    print("Partial indexes created:", partial_count)

    # Check the assignment's minimum partial-index requirement.
    if max_docs is None and partial_count < 3:

        print("WARNING: Fewer than three partial indexes were created.")

    if not document_count:
        raise ValueError(f"No indexable documents found in {dataset_dir}")

    # Find all partial indexes from this build.
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
    generate_report(document_count, unique_token_count, partial_count, skipped_by_reason,
                    index_size_kb, dataset_dir, max_docs, (index_path, documents_path, lexicon_path))

    # Build the compact disk-backed index required by Milestone 3.
    optimize(final_dir)
    write_pagerank(documents_path, links_path, final_dir / "optimized" / "pagerank.bin")


if __name__ == "__main__":
    main()

