"""Build and read a disk-backed field for incoming anchor text."""
import json
import sqlite3
from collections import defaultdict
from contextlib import closing

from pagerank import canonical_url
from tokenizer import process_text


ANCHOR_INDEX = "anchors.sqlite3"
GENERIC_LABELS = frozenset(tuple(process_text(label)) for label in (
    "click", "click here", "here", "learn more", "more", "read more",
    "link", "website", "visit website", "details", "more info",
))


def anchor_terms(labels):
    for label in labels:
        terms = tuple(process_text(label))
        if terms not in GENERIC_LABELS:
            yield from terms


def write_anchors(documents_path, anchors_path, output_path):
    urls = {}
    document_urls = []
    with documents_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["id"] != len(document_urls):
                raise ValueError("Nonconsecutive document IDs in the anchor index.")
            url = canonical_url(record["url"]) or record["url"]
            document_urls.append(urls.setdefault(url, len(urls)))

    with closing(sqlite3.connect(output_path)) as database:
        database.executescript("""
            PRAGMA temp_store = FILE;
            CREATE TEMP TABLE votes (
                term TEXT NOT NULL,
                target INTEGER NOT NULL,
                source INTEGER NOT NULL,
                PRIMARY KEY (term, target, source)
            ) WITHOUT ROWID;
            CREATE TEMP TABLE documents (
                url INTEGER NOT NULL,
                doc INTEGER NOT NULL,
                PRIMARY KEY (url, doc)
            ) WITHOUT ROWID;
            CREATE TABLE postings (
                term TEXT NOT NULL,
                doc INTEGER NOT NULL,
                sources INTEGER NOT NULL,
                PRIMARY KEY (term, doc)
            ) WITHOUT ROWID;
        """)
        with database, anchors_path.open(encoding="utf-8") as rows:
            for source, line in zip(document_urls, rows, strict=True):
                collected = defaultdict(set)
                for target_url, labels in json.loads(line):
                    target = urls.get(target_url)
                    if target is not None and target != source:
                        collected[target].update(anchor_terms(labels))
                database.executemany(
                    "INSERT OR IGNORE INTO votes VALUES (?, ?, ?)",
                    ((term, target, source) for target, terms in collected.items() for term in terms),
                )
            database.executemany(
                "INSERT INTO documents VALUES (?, ?)",
                ((url, doc) for doc, url in enumerate(document_urls)),
            )
            database.execute("""
                INSERT INTO postings
                SELECT grouped.term, documents.doc, grouped.sources
                FROM (
                    SELECT term, target, COUNT(*) AS sources FROM votes GROUP BY term, target
                ) AS grouped
                JOIN documents ON documents.url = grouped.target
            """)
        count = database.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    print(f"Indexed {count:,} incoming anchor postings.")


def open_anchors(path):
    if not path.exists():
        return None
    database = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    try:
        database.execute("SELECT term, doc, sources FROM postings LIMIT 0")
    except sqlite3.Error:
        database.close()
        raise
    return database
