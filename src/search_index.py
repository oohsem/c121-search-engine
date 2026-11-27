"""Read the Boolean search artifacts from one index generation."""
import json
from pathlib import Path
import struct
from index_storage import resolve_index_dir
from functools import lru_cache
from optimize import POSTING, VERSION, bucket_for, source_signature


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


class CompactIndexReader(IndexReader):

    def __init__(self, root=DEFAULT_INDEX_DIR):
        super().__init__(root)

        # Keep only the 16 most recently used lexicon buckets.
        self.load_bucket = lru_cache(maxsize=16)(
            self._load_bucket
        )

    def __enter__(self):

        if self.file is not None:
            raise RuntimeError("The index is already open.")

        try:
            # Locate the currently published index generation.
            directory = resolve_index_dir(self.root)
            optimized = directory / "optimized"

            # Read optimized index metadata.
            manifest = json.loads(
                (optimized / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

            if manifest["version"] != VERSION:
                raise ValueError("Unsupported compact index version.")

            # Make sure the optimized index matches its source.
            signature = source_signature(
                directory / "index.jsonl",
                directory / "documents.jsonl"
            )

            if any(
                manifest.get(key) != value
                for key, value in signature.items()
            ):
                raise ValueError("The compact index is outdated.")

            # Load document IDs and their corresponding URLs.
            doc_urls = {}

            with (directory / "documents.jsonl").open(
                encoding="utf-8"
            ) as documents:

                for line in documents:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    doc_id = record["id"]
                    url = record["url"]

                    if type(doc_id) is not int or doc_id < 0:
                        raise ValueError("Invalid document ID.")

                    if not isinstance(url, str):
                        raise ValueError("Invalid document URL.")

                    if str(doc_id) in doc_urls:
                        raise ValueError("Duplicate document ID.")

                    doc_urls[str(doc_id)] = url

            if len(doc_urls) != manifest["documents"]:
                raise ValueError("Document count does not match.")

            # Load document lengths for relevance normalization.
            length_data = (optimized / "lengths.bin").read_bytes()

            # Each document length occupies 8 bytes.
            if len(length_data) != len(doc_urls) * 8:
                raise ValueError("Invalid document lengths file.")

            lengths = struct.unpack(
                f"<{len(doc_urls)}Q",
                length_data
            )

            average_length = max(
                1.0,
                manifest["average_length"]
)

            # Open the binary index without loading it into memory.
            file = (optimized / "postings.bin").open("rb")

        except (OSError, ValueError, TypeError, KeyError) as error:

            raise IndexArtifactError(
                f"Cannot open compact index: {error}. "
                "Run python src/optimize.py or rebuild the index."
            ) from error

        self.directory = directory
        self.optimized = optimized
        self.manifest = manifest
        self.doc_urls = doc_urls
        self.file = file
        self.lengths = lengths
        self.avg_length = average_length

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        super().__exit__(exc_type, exc_value, traceback)

        self.load_bucket.cache_clear()

    def _load_bucket(self, number):

        bucket = {}

        bucket_path = (
            self.optimized / f"lex_{number:02x}.jsonl"
        )

        with bucket_path.open(encoding="utf-8") as file:

            for line in file:

                term, offset, df = json.loads(line)

                bucket[term] = (offset, df)

        return bucket

    def get_postings(self, term):

        if self.file is None:
            raise RuntimeError(
                "Open the index before reading postings."
            )

        try:

            # Locate the lexicon bucket containing the term.
            bucket = self.load_bucket(bucket_for(term))

            entry = bucket.get(term)

            if entry is None:
                return {}

            offset, df = entry

            # Jump directly to the term's binary postings.
            self.file.seek(offset)

            expected_bytes = df * POSTING.size
            data = self.file.read(expected_bytes)

            if len(data) != expected_bytes:
                raise ValueError("Incomplete binary posting list.")

            postings = {}

            # Decode binary postings into the existing format.
            for doc_id, tf, important_tf in POSTING.iter_unpack(data):

                postings[str(doc_id)] = {
                    "tf": tf,
                    "important_tf": important_tf
                }

            return postings

        except (OSError, ValueError, TypeError, KeyError) as error:

            raise IndexArtifactError(
                f"Cannot retrieve postings for {term}: {error}"
            ) from error