import json


def write_legacy_index(root, documents, terms):
    root.mkdir(parents=True, exist_ok=True)
    offsets = {}
    with (root / "index.jsonl").open("wb") as index:
        for term, postings in sorted(terms.items()):
            offsets[term] = index.tell()
            line = json.dumps({"term": term, "postings": postings}, ensure_ascii=False)
            index.write((line + "\n").encode("utf-8"))
    (root / "lexicon.json").write_text(json.dumps(offsets, ensure_ascii=False), encoding="utf-8")
    with (root / "documents.jsonl").open("w", encoding="utf-8") as document_file:
        for doc_id, url in documents.items():
            document_file.write(json.dumps({"id": doc_id, "url": url}) + "\n")
