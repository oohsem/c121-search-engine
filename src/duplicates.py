"""Group exact and near-duplicate visible content during indexing."""
from array import array
from collections import Counter, defaultdict, deque
from functools import lru_cache
from hashlib import blake2b, sha256
from heapq import heappush, heapreplace, nsmallest
from os import SEEK_END
from sys import byteorder
from unicodedata import normalize

from tokenizer import iter_tokens


DUPLICATE_GROUPS = "duplicate-groups.bin"
HASH_BITS = 64
HAMMING_LIMIT = 3
BAND_BITS = HASH_BITS // (HAMMING_LIMIT + 1)
MIN_WORDS = 30
MIN_DISTINCT_WORDS = 12
SHINGLE_WORDS = 5
SKETCH_SIZE = 256
MIN_SHINGLE_OVERLAP = 0.95
LANE_BITS = 32
LANE_MASK = (1 << LANE_BITS) - 1


def hash64(text):
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


@lru_cache(maxsize=32768)
def word_votes(token):
    hashed = hash64(token)
    # Use independent 32-bit counters packed in one integer. Multiplication
    # weights all 64 positive votes by the term frequency in one operation.
    return sum(1 << (bit * LANE_BITS) for bit in range(HASH_BITS) if hashed & (1 << bit))


def simhash(frequencies, length):
    votes = sum(word_votes(token) * count for token, count in frequencies.items())
    fingerprint = 0
    for bit in range(HASH_BITS):
        if (votes & LANE_MASK) * 2 > length:
            fingerprint |= 1 << bit
        votes >>= LANE_BITS
    return fingerprint


def bands(fingerprint):
    # At most three changed bits leave at least one of four bands unchanged.
    for offset in range(0, HASH_BITS, BAND_BITS):
        yield offset, (fingerprint >> offset) & ((1 << BAND_BITS) - 1)


def shingle_sketch(text):
    """Keep only the lowest hashed distinct shingles, with bounded memory."""
    window = deque(maxlen=SHINGLE_WORDS)
    lowest = []
    sample = set()
    for token in iter_tokens(text):
        window.append(token)
        if len(window) < SHINGLE_WORDS:
            continue
        shingle = hash64(" ".join(window))
        if shingle in sample:
            continue
        if len(lowest) < SKETCH_SIZE:
            heappush(lowest, -shingle)
        elif shingle < -lowest[0]:
            sample.remove(-heapreplace(lowest, -shingle))
        else:
            continue
        sample.add(shingle)
    return frozenset(sample)


def sketch_similarity(left, right):
    union = nsmallest(SKETCH_SIZE, left | right)
    return sum(value in left and value in right for value in union) / len(union) if union else 0.0


class DuplicateDetector:
    """Track canonical fingerprints while leaving postings and links intact."""

    def __init__(self, representative_texts):
        self._exact = {}
        self._representatives = {}
        self._bands = defaultdict(list)
        self._texts = representative_texts
        self.groups = array("I")
        self.exact_count = 0
        self.near_count = 0

        @lru_cache(maxsize=512)
        def representative_sketch(offset, size):
            self._texts.seek(offset)
            return shingle_sketch(self._texts.read(size).decode("utf-8"))

        self._representative_sketch = representative_sketch

    def add(self, text):
        document = len(self.groups)
        digest = sha256(normalize("NFC", " ".join(text.split())).encode("utf-8")).digest()
        canonical = self._exact.get(digest)
        if canonical is not None:
            self.groups.append(canonical)
            self.exact_count += 1
            return

        canonical = self._find_near(text, document)
        self._exact[digest] = canonical
        self.groups.append(canonical)

    def _find_near(self, text, document):
        frequencies = Counter(iter_tokens(text))
        length = sum(frequencies.values())
        if length < MIN_WORDS or len(frequencies) < MIN_DISTINCT_WORDS:
            return document

        fingerprint = simhash(frequencies, length)
        keys = tuple(bands(fingerprint))
        candidates = {candidate for key in keys for candidate in self._bands.get(key, ())}
        close = []
        for candidate in candidates:
            other, other_length, offset, size = self._representatives[candidate]
            if 10 * min(length, other_length) < 9 * max(length, other_length):
                continue
            distance = (fingerprint ^ other).bit_count()
            if distance <= HAMMING_LIMIT:
                close.append((distance, candidate, offset, size))
        if close:
            sketch = shingle_sketch(text)
            for _, candidate, offset, size in sorted(close):
                if sketch_similarity(sketch, self._representative_sketch(offset, size)) >= MIN_SHINGLE_OVERLAP:
                    self.near_count += 1
                    return candidate

        # Compare future pages to representatives only. Similarity is not
        # transitive, so intermediate pages must not join unrelated groups.
        offset = self._texts.seek(0, SEEK_END)
        size = self._texts.write(text.encode("utf-8"))
        self._representatives[document] = (fingerprint, length, offset, size)
        for key in keys:
            self._bands[key].append(document)
        return document

    def write(self, path):
        groups = self.groups[:]
        if byteorder != "little":
            groups.byteswap()
        with path.open("wb") as output:
            groups.tofile(output)


def read_groups(path, count):
    if not path.exists():
        return None
    groups = array("I")
    groups.frombytes(path.read_bytes())
    if byteorder != "little":
        groups.byteswap()
    if len(groups) != count or any(group > doc or groups[group] != group for doc, group in enumerate(groups)):
        raise ValueError("invalid duplicate group table")
    return groups
