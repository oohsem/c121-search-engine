"""Convert the merged M1/M2 JSONL index to compact, seekable M3 files.

Only one term's postings are decoded at a time. The lexicon is partitioned
into 256 deterministic hash buckets, so search need not load it all.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
POSTING = struct.Struct('<III')  # document ID, raw TF, important TF
BUCKETS = 256
BUCKET_BUFFER_BYTES = 16 * 1024
VERSION = 1


def bucket_for(term):
    return zlib.crc32(term.encode('utf-8')) % BUCKETS


def flush_bucket(path, buffer):
    if buffer:
        with path.open("ab") as bucket:
            bucket.write(buffer)
        buffer.clear()


def optimize(index_dir):
    index_dir = Path(index_dir)
    source = index_dir / 'index.jsonl'
    docs = index_dir / 'documents.jsonl'
    if not source.is_file() or not docs.is_file():
        raise FileNotFoundError('Run src/main.py to create the original index first.')
    ids = []
    with docs.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)['id'])
    if not ids or ids != list(range(len(ids))):
        raise ValueError('Expected nonempty documents.jsonl with consecutive numeric IDs.')
    lengths = [0] * len(ids)
    target = index_dir / 'optimized'
    temp = Path(tempfile.mkdtemp(prefix='optimized-build-', dir=index_dir))
    count = 0
    posting_count = 0
    try:
        bucket_paths = [temp / f"lex_{number:02x}.jsonl" for number in range(BUCKETS)]
        for path in bucket_paths:
            path.touch()
        buffers = [bytearray() for _ in range(BUCKETS)]

        # Buffer short writes to each bucket and open at most one bucket at a time.
        with (temp / "postings.bin").open("wb") as out, source.open(encoding="utf-8") as index:
            for line in index:
                record = json.loads(line)
                term, posts = record["term"], record["postings"]
                offset = out.tell()
                data = bytearray()
                for doc, stats in posts.items():
                    doc = int(doc)
                    tf, imp = stats["tf"], stats["important_tf"]
                    data.extend(POSTING.pack(doc, tf, imp))
                    lengths[doc] += tf
                out.write(data)

                number = bucket_for(term)
                buffer = buffers[number]
                lexicon_row = json.dumps([term, offset, len(posts)], ensure_ascii=False, separators=(",", ":"))
                buffer.extend((lexicon_row + "\n").encode("utf-8"))
                if len(buffer) >= BUCKET_BUFFER_BYTES:
                    flush_bucket(bucket_paths[number], buffer)
                count += 1
                posting_count += len(posts)
                if count % 200000 == 0:
                    print(f"Optimized {count:,} terms", flush=True)
            for path, buffer in zip(bucket_paths, buffers):
                flush_bucket(path, buffer)
        # Explicit little-endian values for portability across platforms.
        with (temp / 'lengths.bin').open('wb') as f:
            f.write(struct.pack(f'<{len(lengths)}Q', *lengths))
        manifest = {'version': VERSION, 'documents': len(ids), 'terms': count,
                    'postings': posting_count, 'average_length': sum(lengths) / len(lengths),
                    'source_size': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns,
                    'documents_size': docs.stat().st_size, 'documents_mtime_ns': docs.stat().st_mtime_ns}
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if target.exists():
            shutil.rmtree(target)
        temp.rename(target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    size = sum(p.stat().st_size for p in target.iterdir())
    print(f'Optimized {count:,} terms and {posting_count:,} postings; {size / 1024**2:.2f} MiB.')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, default=PROJECT_DIR / 'final_index')
    optimize(parser.parse_args().index_dir)
