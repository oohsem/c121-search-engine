"""Layout shared by the compact postings writer and reader."""
import struct
import zlib


POSTING = struct.Struct("<III")  # document ID, term frequency, important frequency
BUCKETS = 256
VERSION = 1


def bucket_for(term):
    return zlib.crc32(term.encode("utf-8")) % BUCKETS
