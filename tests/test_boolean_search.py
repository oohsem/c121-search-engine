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
