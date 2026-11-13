from pathlib import Path
import json
from tokenizer import process_text

PROJECT_DIR = Path(__file__).resolve().parent.parent

FINAL_DIR = PROJECT_DIR / "final_index"

INDEX_PATH = FINAL_DIR / "index.jsonl"

DOCUMENTS_PATH = FINAL_DIR / "documents.jsonl"

LEXICON_PATH = FINAL_DIR / "lexicon.json"

#runs once, when this file is imported. Need to remember to close it
with open(LEXICON_PATH, 'r', encoding='utf-8') as file:
    lexicon = json.load(file)

idx_file = open(INDEX_PATH, 'rb')

def load_documents() -> dict[str, str]:
    #returns dict('doc_num', 'url')
    # ex. {'0': 'some_url', '1': 'another_url'}
    doc_urls = dict()
    with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            doc_urls[str(record['id'])] = record['url']

    return doc_urls


def get_postings(term) -> dict[str, dict[str, int]]:
    # returns {'doc_num': {'tf': x, 'important_tf': y}}
    # ex. {'2': {'tf': 5, 'important_tf': 1}, '19': {'tf': 20, 'important_tf': 6}} 
    # obv only returns the docs and its nums with the term
    offset = lexicon.get(term)
    if offset is None:
        return {}

    idx_file.seek(offset)
    line = idx_file.readline()

    return json.loads(line)['postings']

def search(query):
    clean_terms = list(dict.fromkeys(process_text(query))) #list(dict.fromkeys()) removes dupes
    
    if not clean_terms:
        return []

    for term in clean_terms:
        get_postings(term)
    return clean_terms