import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from index_fixture import write_legacy_index
from search_index import IndexArtifactError, IndexReader


class SearchIndexTests(unittest.TestCase):
    def test_reader_closes_the_postings_file_and_can_reopen_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            postings = {"0": {"tf": 1, "important_tf": 0}}
            write_legacy_index(root, {0: "https://example.com/"}, {"café": postings})

            reader = IndexReader(root)
            with reader:
                stream = reader.file
                self.assertEqual(reader.get_postings("café"), postings)
                self.assertEqual(reader.get_postings("absent"), {})
            self.assertTrue(stream.closed)
            with self.assertRaisesRegex(RuntimeError, "Open the index"):
                reader.get_postings("café")
            with reader:
                self.assertEqual(reader.get_postings("café"), postings)

    def test_corrupt_offset_gets_a_clear_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_index(root, {}, {"word": {}})
            (root / "lexicon.json").write_text(json.dumps({"wrong": 0}), encoding="utf-8")

            with IndexReader(root) as reader:
                with self.assertRaisesRegex(IndexArtifactError, "Cannot read postings for 'wrong'"):
                    reader.get_postings("wrong")

    def test_search_cli_explains_missing_artifacts(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "src/search.py", "--index-dir", directory],
                                    capture_output=True, text=True)

            self.assertEqual(result.returncode, 1)
            self.assertIn("Cannot open index at", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
