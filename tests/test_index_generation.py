import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from index_storage import resolve_index_dir
from main import main
from optimize import optimize


class IndexGenerationTests(unittest.TestCase):
    def test_failed_rebuild_keeps_the_last_complete_generation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            page = dataset / "page.json"
            index_root = root / "final_index"
            args = ["--dataset-dir", str(dataset), "--output-root", str(root)]

            self.assertEqual(resolve_index_dir(index_root), index_root)
            self.write_page(page, "original")
            with redirect_stdout(io.StringIO()):
                main(args)
            original = resolve_index_dir(index_root)
            self.assertIn("origin", (original / "index.jsonl").read_text(encoding="utf-8"))
            self.assertTrue((original / "optimized" / "manifest.json").is_file())

            self.write_page(page, "replacement")
            with redirect_stdout(io.StringIO()), patch("main.optimize", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    main(args)
            self.assertEqual(resolve_index_dir(index_root), original)
            self.assertEqual(list((index_root / "generations").iterdir()), [original])

            with redirect_stdout(io.StringIO()):
                main(args)
            current = resolve_index_dir(index_root)
            self.assertNotEqual(current, original)
            self.assertIn("replac", (current / "index.jsonl").read_text(encoding="utf-8"))
            self.assertIn("origin", (original / "index.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(list((root / "partial_indexes").iterdir()), [])

    def test_publication_skips_invalid_destination_urls(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            self.write_page(dataset / "valid.json", "original")
            invalid_urls = ("https://bad host/x", "https://example.org:bad/x",
                            "https://bad\thost/x", "https://bad\nhost/x")
            for number, url in enumerate(invalid_urls):
                (dataset / f"invalid-{number}.json").write_text(
                    json.dumps({"url": url, "content": "not a valid result"}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                main(["--dataset-dir", str(dataset), "--output-root", str(root)])
            current = resolve_index_dir(root / "final_index")
            report = (current / "milestone1_report.txt").read_text(encoding="utf-8")
            self.assertIn("unusable_url: 4", report)
            documents = (current / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(document)["url"] for document in documents],
                             ["https://example.com/"])

    def test_standalone_optimizer_resolves_current_and_preserves_the_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            self.write_page(dataset / "valid.json", "original")
            with redirect_stdout(io.StringIO()):
                main(["--dataset-dir", str(dataset), "--output-root", str(root)])

            index_root = root / "final_index"
            current = resolve_index_dir(index_root)
            target = current / "optimized"
            manifest = target / "manifest.json"
            previous_manifest = manifest.read_bytes()
            old_mtime = manifest.stat().st_mtime_ns

            result = subprocess.run(
                [sys.executable, "-B", "src/optimize.py", "--index-dir", str(index_root)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn(str(target), result.stdout)
            self.assertEqual(manifest.stat().st_mtime_ns, old_mtime)
            self.assertEqual(resolve_index_dir(index_root), current)

            # Simulate a standalone caller encountering a changed flat source.
            legacy_root = root / "legacy"
            legacy_root.mkdir()
            for filename in ("index.jsonl", "documents.jsonl"):
                (legacy_root / filename).write_bytes((current / filename).read_bytes())
            with redirect_stdout(io.StringIO()):
                optimize(legacy_root)
            legacy_manifest = legacy_root / "optimized" / "manifest.json"
            previous_legacy_manifest = legacy_manifest.read_bytes()
            with (legacy_root / "index.jsonl").open("ab") as source:
                source.write(b"\n")
            with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                optimize(legacy_root)
            self.assertEqual(legacy_manifest.read_bytes(), previous_legacy_manifest)
            self.assertEqual(manifest.read_bytes(), previous_manifest)

    @staticmethod
    def write_page(path, text):
        path.write_text(json.dumps({"url": "https://example.com/", "content": text}),
                        encoding="utf-8")
