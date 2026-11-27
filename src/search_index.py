"""Read postings and document metadata from one index generation."""
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

from compact_format import POSTING, VERSION, bucket_for

from index_storage import resolve_index_dir


DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "final_index"


class IndexArtifactError(RuntimeError):
    """The selected index is missing or does not match the expected format."""


class IndexReader:
    def __init__(self, root=DEFAULT_INDEX_DIR):
        self.root = Path(root)
        self.file = None
        self._file_lock = Lock()

    def __enter__(self):
        if self.file is not None:
            raise RuntimeError("The index is already open.")
        try:
            directory = resolve_index_dir(self.root)
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
            compact = directory / "optimized"
            if compact.is_dir():
                manifest = json.loads((compact / "manifest.json").read_text(encoding="utf-8"))
                postings_count = manifest["postings"]
                if (manifest["version"] != VERSION or manifest["documents"] != len(doc_urls)
                        or type(postings_count) is not int or postings_count < 0):
                    raise ValueError("incompatible compact index manifest")
                postings_path = compact / "postings.bin"
                postings_size = postings_path.stat().st_size
                if postings_size != postings_count * POSTING.size:
                    raise ValueError("compact postings file size does not match its manifest")
                lexicon = None
            else:
                compact = None
                postings_size = 0
                postings_path = directory / "index.jsonl"
                lexicon = json.loads((directory / "lexicon.json").read_text(encoding="utf-8"))
                if not isinstance(lexicon, dict) or any(
                    type(offset) is not int or offset < 0 for offset in lexicon.values()
                ):
                    raise ValueError("lexicon must map terms to nonnegative byte offsets")
            file = postings_path.open("rb")
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise IndexArtifactError(
                f"Cannot open index at {self.root}: {error}. Run python src/main.py to build it."
            ) from error

        self.directory = directory
        self.lexicon = lexicon
        self.compact = compact
        self.postings_size = postings_size
        self._bucket = lru_cache(maxsize=32)(self._read_bucket)
        self.doc_urls = doc_urls
        self.file = file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        self.file = None
        self._bucket.cache_clear()

    def get_postings(self, term):
        if self.file is None:
            raise RuntimeError("Open the index before reading postings.")
        if self.compact is not None:
            return {str(doc_id): {"tf": tf, "important_tf": important_tf}
                    for doc_id, tf, important_tf in self.iter_postings(term)}
        offset = self.lexicon.get(term)
        if offset is None:
            return {}

        try:
            with self._file_lock:
                self.file.seek(offset)
                line = self.file.readline()
            record = json.loads(line)
            if record["term"] != term or not isinstance(record["postings"], dict):
                raise ValueError(f"unexpected record at byte offset {offset}")
            return record["postings"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise IndexArtifactError(f"Cannot read postings for {term!r} in {self.directory}: {error}") from error

    def _read_bucket(self, number):
        entries = {}
        path = self.compact / f"lex_{number:02x}.jsonl"
        with path.open(encoding="utf-8") as bucket:
            for line in bucket:
                term, offset, df = json.loads(line)
                if (not isinstance(term, str) or type(offset) is not int or type(df) is not int
                        or offset < 0 or offset % POSTING.size or not 0 < df <= len(self.doc_urls)
                        or offset + df * POSTING.size > self.postings_size):
                    raise ValueError("invalid compact lexicon entry")
                entries[term] = (offset, df)
        return entries

    def posting_range(self, term):
        if self.file is None:
            raise RuntimeError("Open the index before reading postings.")
        if self.compact is None:
            return None
        try:
            return self._bucket(bucket_for(term)).get(term)
        except (OSError, ValueError, TypeError) as error:
            raise IndexArtifactError(f"Cannot look up {term!r} in {self.directory}: {error}") from error

    def iter_postings(self, term, location=None):
        if self.file is None:
            raise RuntimeError("Open the index before reading postings.")
        if self.compact is None:
            return ((int(doc_id), stats["tf"], stats["important_tf"])
                    for doc_id, stats in self.get_postings(term).items())
        location = self.posting_range(term) if location is None else location
        if location is None:
            return iter(())
        offset, df = location
        size = df * POSTING.size
        try:
            with self._file_lock:
                self.file.seek(offset)
                data = self.file.read(size)
            if len(data) != size:
                raise ValueError("truncated postings list")
            return POSTING.iter_unpack(data)
        except (OSError, ValueError) as error:
            raise IndexArtifactError(f"Cannot read postings for {term!r} in {self.directory}: {error}") from error

    def document_url(self, doc_id):
        try:
            return self.doc_urls[doc_id]
        except KeyError as error:
            raise IndexArtifactError(f"Unknown document ID {doc_id} in {self.directory}") from error
