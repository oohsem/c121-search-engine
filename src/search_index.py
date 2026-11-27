"""Read postings and document metadata from one index generation."""
import json
from array import array
from functools import lru_cache
from math import isfinite
from pathlib import Path
from sys import byteorder
from threading import Lock

from compact_format import (
    DOCUMENT_LENGTH,
    NORM_COMPONENTS,
    PAGERANK_SCORE,
    POSTING,
    SUPPORTED_VERSIONS,
    VERSION,
    bucket_for,
    source_signature,
)

from index_storage import resolve_index_dir
from positions import PositionIndex


DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "final_index"
POSTINGS_PER_READ = 4096


class IndexArtifactError(RuntimeError):
    """The selected index is missing or does not match the expected format."""


class IndexReader:
    def __init__(self, root=DEFAULT_INDEX_DIR, *, include_sources=False):
        self.root = Path(root)
        self.include_sources = include_sources
        self.file = None
        self._file_lock = Lock()

    def __enter__(self):
        if self.file is not None:
            raise RuntimeError("The index is already open.")
        try:
            directory = resolve_index_dir(self.root)
            doc_urls = {}
            doc_sources = {}
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
                    if self.include_sources and isinstance(record.get("doc_id"), str):
                        doc_sources[doc_id] = record["doc_id"]
            compact = directory / "optimized"
            if compact.is_dir():
                manifest = json.loads((compact / "manifest.json").read_text(encoding="utf-8"))
                compact_version = manifest["version"]
                postings_count = manifest["postings"]
                original = directory / "index.jsonl"
                if original.is_file() and any(
                    manifest.get(key) != value
                    for key, value in source_signature(original, directory / "documents.jsonl").items()
                ):
                    raise ValueError("the compact index is outdated")
                if (compact_version not in SUPPORTED_VERSIONS or manifest["documents"] != len(doc_urls)
                        or type(postings_count) is not int or postings_count < 0):
                    raise ValueError("incompatible compact index manifest")
                postings_path = compact / "postings.bin"
                postings_size = postings_path.stat().st_size
                if postings_size != postings_count * POSTING.size:
                    raise ValueError("compact postings file size does not match its manifest")
                lexicon = None
            else:
                compact = None
                compact_version = 0
                postings_size = 0
                postings_path = directory / "index.jsonl"
                lexicon = json.loads((directory / "lexicon.json").read_text(encoding="utf-8"))
                if not isinstance(lexicon, dict) or any(
                    type(offset) is not int or offset < 0 for offset in lexicon.values()
                ):
                    raise ValueError("lexicon must map terms to nonnegative byte offsets")
            file = postings_path.open("rb")
            try:
                positions = PositionIndex.open(directory, file, postings_count) if compact is not None else None
            except BaseException:
                file.close()
                raise
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise IndexArtifactError(
                f"Cannot open index at {self.root}: {error}. Run python src/main.py to build it."
            ) from error

        self.directory = directory
        self.lexicon = lexicon
        self.manifest = manifest if compact is not None else {}
        self.compact = compact
        self.compact_version = compact_version
        self.postings_size = postings_size
        self._bucket = lru_cache(maxsize=32)(self._read_bucket)
        self.doc_urls = doc_urls
        self.doc_sources = doc_sources
        self.file = file
        self.positions = positions
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.positions is not None:
            self.positions.close()
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
        return self._compact_postings(term, *location)

    def _compact_postings(self, term, offset, df):
        try:
            for start in range(0, df, POSTINGS_PER_READ):
                size = min(POSTINGS_PER_READ, df - start) * POSTING.size
                with self._file_lock:
                    self.file.seek(offset + start * POSTING.size)
                    data = self.file.read(size)
                if len(data) != size:
                    raise ValueError("truncated postings list")
                yield from POSTING.iter_unpack(data)
        except (OSError, ValueError) as error:
            raise IndexArtifactError(f"Cannot read postings for {term!r} in {self.directory}: {error}") from error

    def norm_components(self):
        if self.file is None:
            raise RuntimeError("Open the index before loading document norms.")
        if self.compact_version != VERSION:
            raise IndexArtifactError("Ranked search needs document norms. Run python src/main.py to rebuild the index.")
        try:
            data = (self.compact / "norms.bin").read_bytes()
            if len(data) != len(self.doc_urls) * NORM_COMPONENTS.size:
                raise ValueError("norm file size does not match the document count")
            return NORM_COMPONENTS.iter_unpack(data)
        except (OSError, ValueError) as error:
            raise IndexArtifactError(f"Cannot read document norms in {self.directory}: {error}") from error

    def document_lengths(self):
        if self.file is None or self.compact is None:
            raise IndexArtifactError("Open a compact index before loading document lengths.")
        try:
            data = (self.compact / "lengths.bin").read_bytes()
            if len(data) != len(self.doc_urls) * DOCUMENT_LENGTH.size:
                raise ValueError("length file size does not match the document count")
            return (length for length, in DOCUMENT_LENGTH.iter_unpack(data))
        except (OSError, ValueError) as error:
            raise IndexArtifactError(f"Cannot read document lengths in {self.directory}: {error}") from error

    def pagerank_scores(self):
        if self.file is None or self.compact is None:
            return None
        path = self.compact / "pagerank.bin"
        if not path.exists():
            return None
        try:
            with path.open("rb") as file:
                scores = array("d")
                if path.stat().st_size != len(self.doc_urls) * PAGERANK_SCORE.size:
                    raise ValueError("score file size does not match the document count")
                scores.fromfile(file, len(self.doc_urls))
            if byteorder != "little":
                scores.byteswap()
            if any(not isfinite(score) or score < 0 for score in scores):
                raise ValueError("invalid PageRank score")
            return scores
        except (OSError, EOFError, ValueError) as error:
            raise IndexArtifactError(f"Cannot read PageRank in {self.directory}: {error}") from error

    def document_url(self, doc_id):
        try:
            return self.doc_urls[doc_id]
        except KeyError as error:
            raise IndexArtifactError(f"Unknown document ID {doc_id} in {self.directory}") from error



class CompactIndexReader(IndexReader):
    """Compatibility entry point for callers requiring compact postings."""

    def __enter__(self):
        super().__enter__()
        try:
            self.lengths = tuple(self.document_lengths())
            self.avg_length = max(1.0, self.manifest["average_length"])
        except (IndexArtifactError, KeyError, TypeError):
            self.__exit__(None, None, None)
            raise IndexArtifactError("Cannot open compact postings. Rebuild the index.") from None
        return self
