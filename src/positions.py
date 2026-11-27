"""Disk-backed token positions and positional query matching."""
import re
from base64 import b64decode, b64encode
from collections import deque
from contextlib import ExitStack, contextmanager
from heapq import merge
from itertools import accumulate
from mmap import ACCESS_READ, mmap
from struct import Struct

from compact_format import POSTING
from tokenizer import iter_tokens, process_text, stem_word


POSITION_OFFSETS = "position-offsets.bin"
POSITION_DATA = "positions.bin"
OFFSET = Struct("<Q")
FAST_DECODE_BYTES = 4096
QUOTED = re.compile(r'"([^"]*)"')


def position_terms(text):
    """Count each stem and encode its increasing token offsets as gaps."""
    entries = {}
    for position, token in enumerate(iter_tokens(text)):
        term = stem_word(token)
        entry = entries.get(term)
        if entry is None:
            entry = entries[term] = [0, 0, bytearray()]
        gap = position - entry[1]
        entry[0] += 1
        entry[1] = position
        data = entry[2]
        while gap >= 128:
            data.append((gap & 127) | 128)
            gap >>= 7
        data.append(gap)

    return ((term, count, b64encode(data).decode("ascii"))
            for term, (count, _, data) in entries.items())


@contextmanager
def position_writer(directory):
    """Write one position range per compact posting, in posting order."""
    with (directory / POSITION_OFFSETS).open("wb") as offsets, (directory / POSITION_DATA).open("wb") as data:
        offsets.write(OFFSET.pack(0))

        def append(encoded):
            data.write(b64decode(encoded, validate=True))
            offsets.write(OFFSET.pack(data.tell()))

        yield append


class PositionIndex:
    @classmethod
    def open(cls, directory, postings, posting_count):
        offsets_path = directory / POSITION_OFFSETS
        data_path = directory / POSITION_DATA
        if not offsets_path.exists() and not data_path.exists():
            return None

        with ExitStack() as files:
            offsets_file = files.enter_context(offsets_path.open("rb"))
            data_file = files.enter_context(data_path.open("rb"))
            if offsets_path.stat().st_size != (posting_count + 1) * OFFSET.size:
                raise ValueError("position offset count does not match the postings file")

            offsets = files.enter_context(mmap(offsets_file.fileno(), 0, access=ACCESS_READ))
            size = data_path.stat().st_size
            if OFFSET.unpack_from(offsets)[0] != 0 or OFFSET.unpack_from(offsets, posting_count * OFFSET.size)[0] != size:
                raise ValueError("position offsets do not match the position data")

            index = cls()
            index.offsets = offsets
            index.data = files.enter_context(mmap(data_file.fileno(), 0, access=ACCESS_READ)) if size else b""
            index.postings = files.enter_context(mmap(postings.fileno(), 0, access=ACCESS_READ)) if posting_count else b""
            index.files = files.pop_all()
            return index

    def close(self):
        self.files.close()

    def rows(self, doc_id, locations, counts, cursors=None):
        """Find each candidate's rows; reuse cursors for increasing document IDs."""
        rows = {}
        for term, (offset, df) in locations.items():
            if cursors is None:
                low, high = 0, df
                while low < high:
                    middle = (low + high) // 2
                    other, _, _ = POSTING.unpack_from(self.postings, offset + middle * POSTING.size)
                    if other < doc_id:
                        low = middle + 1
                    else:
                        high = middle
                if low == df:
                    return None
                other, frequency, _ = POSTING.unpack_from(self.postings, offset + low * POSTING.size)
            else:
                low = cursors[term]
                while low < df:
                    other, frequency, _ = POSTING.unpack_from(self.postings, offset + low * POSTING.size)
                    if other >= doc_id:
                        break
                    low += 1
                else:
                    return None
                cursors[term] = low + (other == doc_id)
            if other != doc_id or frequency < counts[term]:
                return None
            rows[term] = offset // POSTING.size + low
        return rows

    def values(self, row):
        return self._values(*self._range(row))

    def _range(self, row):
        start = OFFSET.unpack_from(self.offsets, row * OFFSET.size)[0]
        end = OFFSET.unpack_from(self.offsets, (row + 1) * OFFSET.size)[0]
        if not start < end <= len(self.data):
            raise ValueError("invalid position range")
        return start, end

    def _values(self, start, end):
        # Common terms often have short gaps. Keep the fast path bounded,
        # and stream the variable-length row when it cannot use that path.
        if end - start <= FAST_DECODE_BYTES:
            gaps = self.data[start:end]
            if gaps.isascii():
                return accumulate(gaps)
        return decode_positions(self.data, start, end)

    def matches_phrase(self, phrase, rows):
        ranges = {term: self._range(rows[term]) for _, term in phrase}
        sizes = {term: end - start for term, (start, end) in ranges.items()}

        def values(term):
            return self._values(*ranges[term])

        if sum(sizes.values()) <= FAST_DECODE_BYTES:
            # Short rows fit in a small per-document buffer. Reuse a term's
            # positions for every offset where it occurs in the phrase.
            plan = sorted(phrase, key=lambda item: sizes[item[1]])
            return contains_short_phrase(plan, values)
        return contains_phrase(phrase, values)


def decode_positions(data, start, end):
    position = gap = shift = 0
    for offset in range(start, end):
        value = data[offset]
        gap |= (value & 127) << shift
        if value & 128:
            shift += 7
            if shift > 28:
                raise ValueError("position gap is too large")
            continue
        position += gap
        yield position
        gap = shift = 0
    if shift:
        raise ValueError("truncated position gap")


def quoted_phrases(query):
    return [terms for match in QUOTED.finditer(query) if (terms := process_text(match[1]))]


def contains_short_phrase(phrase, positions):
    anchor_offset, anchor_term = phrase[0]
    anchor = set(positions(anchor_term))
    cached = {anchor_term: anchor}
    for offset, term in phrase[1:]:
        if term not in cached:
            cached[term] = set(positions(term))
        other = cached[term]
        shift = offset - anchor_offset
        anchor = {position for position in anchor if position + shift in other}
        if not anchor:
            return False
    return bool(anchor)


def contains_phrase(phrase, positions):
    # Intersect positional streams with the phrase offsets applied. If a
    # stream jumps ahead, no intervening anchor position can match.
    anchor_offset, anchor_term = phrase[0]
    anchor = positions(anchor_term)
    others = [(offset - anchor_offset, positions(term)) for offset, term in phrase[1:]]
    current = [next(stream, None) for _, stream in others]
    candidate = next(anchor, None)
    while candidate is not None:
        for index, (offset, stream) in enumerate(others):
            target = candidate + offset
            position = current[index]
            while position is not None and position < target:
                position = next(stream, None)
            if position is None:
                return False
            current[index] = position
            if position > target:
                target = position - offset
                while candidate is not None and candidate < target:
                    candidate = next(anchor, None)
                break
        else:
            return True
    return False


def events(term, positions):
    for position in positions:
        yield position, term


def smallest_window(counts, positions):
    """Return the shortest window covering the requested term counts."""
    recent = {term: deque(maxlen=count) for term, count in counts.items()}
    required = sum(counts.values())
    best = None
    missing = len(counts)
    for end, term in merge(*(events(term, positions(term)) for term in counts)):
        queue = recent[term]
        first_complete = len(queue) == queue.maxlen - 1
        queue.append(end)
        if first_complete:
            missing -= 1
        if missing:
            continue
        span = end - min(queue[0] for queue in recent.values()) + 1
        if best is None or span < best:
            best = span
            if best == required:
                break
    return best
