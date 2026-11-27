import unittest

from parser import parse_html


class ParseHtmlTests(unittest.TestCase):
    def test_declarations_and_nonvisible_nodes_are_not_indexed(self):
        html = ('<!DOCTYPE html><?xml version="1.0"?>'
                '<!-- hidden --><script>javascript</script>'
                '<style>css</style><p>Visible text</p>')
        self.assertEqual(parse_html(html), ("Visible text", ""))

    def test_plain_text_and_empty_content(self):
        self.assertEqual(parse_html("Plain page text"), ("Plain page text", ""))
        self.assertEqual(parse_html(""), ("", ""))

    def test_unclosed_important_tags(self):
        self.assertEqual(parse_html("<h1>A heading <b>with bold"),
                         ("A heading with bold", "A heading with bold"))
