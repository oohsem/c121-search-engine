"""Read the Boolean search artifacts from one index generation."""
import json
from pathlib import Path

from index_storage import resolve_index_dir


DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "final_index"


class IndexArtifactError(RuntimeError):
    """The selected index is missing or does not match the expected format."""


class IndexReader:
    def __init__(self, root=DEFAULT_INDEX_DIR):
        self.root = Path(root)
        self.file = None

    def __enter__(self):
        if self.file is not None:
            raise RuntimeError("The index is already open.")
        try:
            directory = resolve_index_dir(self.root)
            lexicon = json.loads((directory / "lexicon.json").read_text(encoding="utf-8"))
            if not isinstance(lexicon, dict) or any(
                type(offset) is not int or offset < 0 for offset in lexicon.values()
            ):
                raise ValueError("lexicon must map terms to nonnegative byte offsets")
            doc_urls = {}
            with (directory / "documents.jsonl").open(encoding="utf-8") as documents:
                for line in documents:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    doc_id, url = record["id"], record["url"]
                    if type(doc_id) is not int or doc_id < 0 or not isinstance(url, str):
                        raise ValueError("invalid document ID or URL")
                    if str(doc_id) in doc_urls:
                        raise ValueError(f"duplicate document ID {doc_id}")
                    doc_urls[str(doc_id)] = url
            file = (directory / "index.jsonl").open("rb")
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise IndexArtifactError(
                f"Cannot open index at {self.root}: {error}. Run python src/main.py to build it."
            ) from error

        self.directory = directory
        self.lexicon = lexicon
        self.doc_urls = doc_urls
        self.file = file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        self.file = None

    def get_postings(self, term):
        if self.file is None:
            raise RuntimeError("Open the index before reading postings.")
        offset = self.lexicon.get(term)
        if offset is None:
            return {}

        try:
            self.file.seek(offset)
            record = json.loads(self.file.readline())
            if record["term"] != term or not isinstance(record["postings"], dict):
                raise ValueError(f"unexpected record at byte offset {offset}")
            return record["postings"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise IndexArtifactError(f"Cannot read postings for {term!r} in {self.directory}: {error}") from error

    def document_url(self, doc_id):
        try:
            return self.doc_urls[doc_id]
        except KeyError as error:
            raise IndexArtifactError(f"Unknown document ID {doc_id} in {self.directory}") from error
