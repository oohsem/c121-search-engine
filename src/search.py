from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_DIR / "final_index"

INDEX_PATH = FINAL_DIR / "index.jsonl"

DOCUMENTS_PATH = FINAL_DIR / "documents.jsonl"

def load_documents():
    doc_urls = dict()
    with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            doc_urls[record['doc_id']] = record['url']

    return doc_urls


def get_postings():

    idx_path = Path(INDEX_PATH)
    with open(idx_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]

    return

def search(query):
    return