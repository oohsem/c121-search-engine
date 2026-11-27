"""Convert the merged M1/M2 JSONL index to compact, seekable M3 files.

Only one term's postings are decoded at a time. The lexicon is partitioned
into 256 deterministic hash buckets, so search need not load it all.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

from compact_format import BUCKETS, NORM_COMPONENTS, POSTING, SUPPORTED_VERSIONS, VERSION, bucket_for, lexicon_row
from index_storage import resolve_index_dir
from ranking import document_weights

PROJECT_DIR = Path(__file__).resolve().parent.parent
BUCKET_BUFFER_BYTES = 16 * 1024


def flush_bucket(path, buffer):
    if buffer:
        with path.open("ab") as bucket:
            bucket.write(buffer)
        buffer.clear()


def source_signature(postings, documents):
    postings_stat = postings.stat()
    documents_stat = documents.stat()
    return {"source_size": postings_stat.st_size, "source_mtime_ns": postings_stat.st_mtime_ns,
            "documents_size": documents_stat.st_size, "documents_mtime_ns": documents_stat.st_mtime_ns}


def optimize(index_dir):
    index_dir = Path(index_dir)
    source = index_dir / 'index.jsonl'
    docs = index_dir / 'documents.jsonl'
    if not source.is_file() or not docs.is_file():
        raise FileNotFoundError('Run src/main.py to create the original index first.')
    signature = source_signature(source, docs)
    target = index_dir / "optimized"
    if target.exists():
        try:
            existing = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if (isinstance(existing, dict) and existing.get("version") == VERSION
                and all(existing.get(key) == value for key, value in signature.items())
                and all((target / name).is_file() for name in ("postings.bin", "lengths.bin", "norms.bin"))
                and all((target / f"lex_{number:02x}.jsonl").is_file() for number in range(BUCKETS))):
            print(f"Compact index already available at {target}.")
            return existing
        raise FileExistsError(f"Refusing to replace the existing compact index at {target}. "
                              "Run src/main.py to publish a fresh generation.")

    ids = []
    with docs.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)['id'])
    if not ids or ids != list(range(len(ids))):
        raise ValueError('Expected nonempty documents.jsonl with consecutive numeric IDs.')
    lengths = [0] * len(ids)
    base_squared = [0.0] * len(ids)
    cross = [0.0] * len(ids)
    emphasis_squared = [0.0] * len(ids)
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
                    base, emphasis = document_weights(tf, imp)
                    base_squared[doc] += base * base
                    cross[doc] += base * emphasis
                    emphasis_squared[doc] += emphasis * emphasis
                out.write(data)

                number = bucket_for(term)
                buffer = buffers[number]
                buffer.extend(lexicon_row(term, offset, len(posts)))
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
        with (temp / "norms.bin").open("wb") as norms:
            for components in zip(base_squared, cross, emphasis_squared):
                norms.write(NORM_COMPONENTS.pack(*components))
        if source_signature(source, docs) != signature:
            raise RuntimeError("Index sources changed during optimization. Retry on a complete build.")
        manifest = {'version': VERSION, 'documents': len(ids), 'terms': count,
                    'postings': posting_count, 'average_length': sum(lengths) / len(lengths),
                    **signature}
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temp.rename(target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    size = sum(p.stat().st_size for p in target.iterdir())
    print(f'Optimized {count:,} terms and {posting_count:,} postings; {size / 1024**2:.2f} MiB.')
    return manifest


def verify(index_dir):
    source = index_dir / "index.jsonl"
    compact = index_dir / "optimized"
    manifest = json.loads((compact / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] not in SUPPORTED_VERSIONS:
        raise ValueError("Cannot verify an unsupported compact index version.")
    if manifest["version"] == VERSION and (
        (compact / "norms.bin").stat().st_size != manifest["documents"] * NORM_COMPONENTS.size
    ):
        raise ValueError("Document norm file size does not match the manifest.")

    expected_buckets = [hashlib.sha256() for _ in range(BUCKETS)]
    terms = posting_count = 0
    with source.open(encoding="utf-8") as original, (compact / "postings.bin").open("rb") as binary:
        for line in original:
            record = json.loads(line)
            term, postings = record["term"], record["postings"]
            offset = binary.tell()
            expected = bytearray()
            for doc, stats in postings.items():
                expected.extend(POSTING.pack(int(doc), stats["tf"], stats["important_tf"]))
            if binary.read(len(expected)) != expected:
                raise ValueError(f"Compact postings differ from JSONL for {term!r}.")
            expected_buckets[bucket_for(term)].update(lexicon_row(term, offset, len(postings)))
            terms += 1
            posting_count += len(postings)
        if binary.read(1) or terms != manifest["terms"] or posting_count != manifest["postings"]:
            raise ValueError("Compact postings totals differ from the JSONL source or manifest.")

    for number, expected in enumerate(expected_buckets):
        with (compact / f"lex_{number:02x}.jsonl").open("rb") as bucket:
            if hashlib.file_digest(bucket, "sha256").digest() != expected.digest():
                raise ValueError(f"Compact lexicon bucket {number:02x} mismatches the JSONL source.")
    print(f"Verified {terms:,} terms and {posting_count:,} postings against JSONL.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, default=PROJECT_DIR / 'final_index')
    parser.add_argument("--verify", action="store_true", help="Compare compact postings and lexicon against JSONL.")
    args = parser.parse_args()
    try:
        directory = resolve_index_dir(args.index_dir)
        if args.verify:
            verify(directory)
        else:
            optimize(directory)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
