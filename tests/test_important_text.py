import unittest

from indexer import build_index
from parser import parse_html


class ImportantTextTests(unittest.TestCase):
    def test_required_tags_and_nested_tags_count_each_occurrence_once(self):
        tags = ("title", "h1", "h2", "h3", "b", "strong")
        html = "".join(f"<{tag}>word</{tag}>" for tag in tags)
        html += "<h1><strong>word</strong></h1><p>word</p>"
        index = {}

        build_index(*parse_html(html), 0, index)

        self.assertEqual(index["word"][0], {"tf": 8, "important_tf": 7})
