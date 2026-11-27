import unittest

from indexer import build_index
from tokenizer import process_text, tokenize


class TokenizerTests(unittest.TestCase):
    def test_alphanumeric_sequences_and_case(self):
        self.assertEqual(tokenize("ABC123, CAFÉ42! x-y foo_bar 42."),
                         ["abc123", "café42", "x", "y", "foo", "bar", "42"])

    def test_index_and_query_use_the_same_stems(self):
        index = {}
        build_index("The runners are running and training", "", 0, index)
        self.assertEqual(process_text("RUNNING trains"), ["run", "train"])
        self.assertTrue(all(term in index for term in process_text("RUNNING trains")))
        self.assertTrue({"the", "and", "are"}.issubset(index))
