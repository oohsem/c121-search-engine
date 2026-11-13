from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_DIR / "final_index"

INDEX_PATH = FINAL_DIR / "index.jsonl"

DOCUMENTS_PATH = FINAL_DIR / "documents.jsonl"

def load_documents():
    doc_path = Path(DOCUMENTS_PATH)

    with open(doc_path, 'r', encoding='utf-8') as file:
        doc_urls = [json.loads(line) for line in file if line.strip()]

    return doc_urls


def get_postings():

    idx_path = Path(INDEX_PATH)
    with open(idx_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]

    return

def search(query):
    return