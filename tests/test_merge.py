import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from indexer import merge_partial_indexes, save_partial_index


class MergeTests(unittest.TestCase):
    def test_spilled_records_and_utf8_offsets_survive_the_merge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [
                {"zulu": {0: {"tf": 1, "important_tf": 0}},
                 "café": {0: {"tf": 2, "important_tf": 1}}},
                {"café": {1: {"tf": 1, "important_tf": 0}},
                 "marker": {1: {"tf": 3, "important_tf": 2}}},
                {"marker": {2: {"tf": 1, "important_tf": 0}},
                 "café": {2: {"tf": 4, "important_tf": 2}}},
            ]
            spills = [save_partial_index(run, root, number)
                      for number, run in enumerate(runs, 1)]
            self.assertTrue(all(not run for run in runs))
            index_path = root / "index.jsonl"
            lexicon_path = root / "lexicon.json"

            self.assertEqual(merge_partial_indexes(spills, index_path, lexicon_path), 3)

            lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
            self.assertEqual(list(lexicon), ["café", "marker", "zulu"])
            expected = {
                "café": {"0": {"tf": 2, "important_tf": 1},
                         "1": {"tf": 1, "important_tf": 0},
                         "2": {"tf": 4, "important_tf": 2}},
                "marker": {"1": {"tf": 3, "important_tf": 2},
                           "2": {"tf": 1, "important_tf": 0}},
                "zulu": {"0": {"tf": 1, "important_tf": 0}},
            }
            with index_path.open("rb") as index:
                for term, offset in lexicon.items():
                    index.seek(offset)
                    self.assertEqual(json.loads(index.readline()),
                                     {"term": term, "postings": expected[term]})
