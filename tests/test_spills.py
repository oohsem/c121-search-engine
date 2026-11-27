import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from main import main


class SpillBudgetTests(unittest.TestCase):
    def test_budget_triggers_a_spill_at_each_document_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            for number in range(4):
                (dataset / f"{number}.json").write_text(json.dumps({
                    "url": f"https://example.com/{number}",
                    "content": "one two three",
                }), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                main(["--dataset-dir", str(dataset), "--output-root", str(root),
                      "--max-postings", "2"])

            spills = sorted((root / "partial_indexes").glob("partial_*.jsonl"))
            self.assertEqual(len(spills), 4)
            for spill in spills:
                rows = [json.loads(line) for line in spill.read_text(encoding="utf-8").splitlines()]
                self.assertEqual({row["term"] for row in rows}, {"one", "two", "three"})
                self.assertEqual(len({doc for row in rows for doc in row["postings"]}), 1)
            self.assertIn("Writing partial index 4: 3 postings", output.getvalue())
