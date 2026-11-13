from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_DIR / "final_index"

INDEX_PATH = FINAL_DIR / "index.jsonl"

DOCUMENTS_PATH = FINAL_DIR / "documents.jsonl"

LEXICON_PATH = FINAL_DIR / "lexicon.json"

#runs once, when this file is imported. Need to remember to close it
with open(LEXICON_PATH, 'r', encoding='utf-8') as file:
    lexicon = json.load(file)

idx_file = open(INDEX_PATH, 'rb')

def load_documents():
    doc_urls = dict()
    with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            doc_urls[str(record['id'])] = record['url']

    return doc_urls


def get_postings(term):
    offset = lexicon.get(term)
    if offset is None:
        return {}

    idx_file.seek(offset)
    line = idx_file.readline()

    return json.loads(line)['postings']

def search(query):
    return