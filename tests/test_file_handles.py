import subprocess
import sys
import textwrap
import unittest


@unittest.skipIf(sys.platform == "win32", "requires a POSIX descriptor limit")
class FileHandleTests(unittest.TestCase):
    def test_merge_and_optimizer_run_under_a_small_descriptor_limit(self):
        script = textwrap.dedent("""
            import json
            import resource
            from pathlib import Path
            from tempfile import TemporaryDirectory

            from indexer import merge_partial_indexes, save_partial_index
            from optimize import POSTING, bucket_for, optimize

            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft_limit < 0 or soft_limit > 32:
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, hard_limit))
            with TemporaryDirectory() as directory:
                root = Path(directory)
                runs = []
                for doc in range(64):
                    postings = {doc: {"tf": doc + 1, "important_tf": 1}}
                    terms = {"café": postings, f"term{doc}": postings}
                    runs.append(save_partial_index(terms, root, doc))
                (root / "documents.jsonl").write_text("".join(
                    json.dumps({"id": doc}) + "\\n" for doc in range(64)
                ), encoding="utf-8")
                assert merge_partial_indexes(runs, root / "index.jsonl", root / "lexicon.json") == 65
                assert all(run.is_file() for run in runs)
                manifest = optimize(root)
                assert manifest["terms"] == 65 and manifest["postings"] == 128
                optimized = root / "optimized"
                assert len(list(optimized.glob("lex_*.jsonl"))) == 256
                bucket = optimized / f"lex_{bucket_for('café'):02x}.jsonl"
                rows = {row[0]: row[1:] for row in map(json.loads, bucket.read_text().splitlines())}
                offset, count = rows["café"]
                assert count == 64
                with (optimized / "postings.bin").open("rb") as file:
                    file.seek(offset)
                    records = list(POSTING.iter_unpack(file.read(count * POSTING.size)))
                assert records == [(doc, doc + 1, 1) for doc in range(64)], records
        """)
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
