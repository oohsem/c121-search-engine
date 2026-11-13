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

doc_urls = load_documents()

def search(query, top_k=5):
    clean_terms = list(dict.fromkeys(process_text(query))) #list(dict.fromkeys()) removes dupes
    
    if not clean_terms:
        return []

    all_posts = []
    for term in clean_terms:
        postings = get_postings(term)
        if not postings:
            return []
        all_posts.append(postings)

    all_posts.sort(key=len)
    matches = set(all_posts[0])
    for posts in all_posts[1:]:
        matches &= set(posts)
        if not matches: 
            return [] 

    
    urls = [doc_urls[i] for i in matches] # dont use get, use [i] because we need it to fail loudly to bugtest, once we are done, we revert back to get()
    return urls[:top_k]

if __name__ == '__main__':
    while True:
        query = input("Give Query: ").strip()
        if not query:
            break

        results = search(query)

        for rank, url in enumerate(results):
            print(rank, '. ', url)