"""Layout shared by the compact postings writer and reader."""
import json
import struct
import zlib


POSTING = struct.Struct("<III")  # document ID, term frequency, important frequency
BUCKETS = 256
# Base squared, base times emphasis, and emphasis squared for each document.
NORM_COMPONENTS = struct.Struct("<ddd")
DOCUMENT_LENGTH = struct.Struct("<Q")
VERSION = 2
SUPPORTED_VERSIONS = (1, VERSION)


def bucket_for(term):
    return zlib.crc32(term.encode("utf-8")) % BUCKETS


def lexicon_row(term, offset, df):
    row = json.dumps([term, offset, df], ensure_ascii=False, separators=(",", ":"))
    return (row + "\n").encode("utf-8")
