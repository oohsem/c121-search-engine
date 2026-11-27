import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from baseline_search import search as baseline_search
from index_fixture import write_legacy_index
from search import search
from search_index import IndexReader


class BooleanSearchTests(unittest.TestCase):
    def test_stemmed_plain_text_is_a_conjunction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = "https://example.com/first"
            second = "https://example.com/second"
            stats = {"tf": 1, "important_tf": 0}
            write_legacy_index(root, {0: first, 1: second},
                               {"run": {"0": stats, "1": stats}, "train": {"1": stats}})

            self.assertIs(baseline_search, search)
            with IndexReader(root) as index:
                self.assertEqual(search("running trains RUN", index=index), [second])
                self.assertEqual(search("running missing", index=index), [])
                self.assertEqual(search("!!!", index=index), [])

            for script in ("src/search.py", "src/baseline_search.py"):
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, "-B", script, "--index-dir", str(root)],
                                            input="running trains\n\n", capture_output=True,
                                            text=True, check=True)
                    self.assertIn(second, result.stdout)
                    self.assertNotIn(first, result.stdout)

    def test_top_five_do_not_depend_on_the_hash_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ids = (10, 41, 30, 2, 20, 1)
            urls = {doc_id: f"https://example.com/{doc_id}" for doc_id in ids}
            write_legacy_index(root, urls,
                               {"sampl": {str(doc_id): {"tf": 1, "important_tf": 0}
                                          for doc_id in ids}})
            expected = [urls[doc_id] for doc_id in (1, 2, 10, 20, 30)]

            for seed in ("0", "1", "2"):
                result = subprocess.run(
                    [sys.executable, "-B", "src/search.py", "--index-dir", str(root)],
                    env={**os.environ, "PYTHONHASHSEED": seed},
                    input="samples\n\n", text=True, capture_output=True, check=True,
                )
                actual = [line.split()[-1] for line in result.stdout.splitlines()
                          if "https://" in line]
                self.assertEqual(actual, expected, f"hash seed {seed}")
