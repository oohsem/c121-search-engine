import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from indexer import save_partial_index
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

            spills = []

            def observe_spill(index, partial_dir, number):
                path = save_partial_index(index, partial_dir, number)
                spills.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
                return path

            output = io.StringIO()
            with redirect_stdout(output), patch("main.save_partial_index", side_effect=observe_spill):
                main(["--dataset-dir", str(dataset), "--output-root", str(root),
                      "--max-postings", "2"])

            self.assertEqual(len(spills), 4)
            for rows in spills:
                self.assertEqual({row["term"] for row in rows}, {"one", "two", "three"})
                self.assertEqual(len({doc for row in rows for doc in row["postings"]}), 1)
            self.assertIn("Writing partial index 4: 3 postings", output.getvalue())
            self.assertEqual(list((root / "partial_indexes").iterdir()), [])
