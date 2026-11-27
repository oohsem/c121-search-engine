import warnings
from html.parser import HTMLParser

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from bs4.element import Comment, Declaration, Doctype, ProcessingInstruction


IMPORTANT_TAGS = {"title", "h1", "h2", "h3", "b", "strong"}

IGNORED_TAGS = {"script", "style", "noscript", "template"}
LINK_TAGS = ("a", "area", "base")


LARGE_HTML_CHARS = 8 * 1024 * 1024
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
             "meta", "param", "source", "track", "wbr"}


class LargePageText(HTMLParser):
    """Collect searchable text without holding a tree for a large page."""

    def __init__(self, on_link=None):
        super().__init__(convert_charrefs=True)
        self.on_link = on_link
        self.tags = []
        self.ignored_depth = 0
        self.important_depth = 0
        self.visible = []
        self.important = []

    def handle_starttag(self, tag, attrs):
        if self.on_link is not None and not self.ignored_depth and tag in LINK_TAGS:
            href = next((value for name, value in attrs if name == "href"), None)
            if href is not None:
                self.on_link(tag, href)
        if tag not in VOID_TAGS:
            self.tags.append(tag)
            self.ignored_depth += tag in IGNORED_TAGS
            self.important_depth += tag in IMPORTANT_TAGS

    def handle_endtag(self, tag):
        for position in range(len(self.tags) - 1, -1, -1):
            if self.tags[position] == tag:
                for removed in self.tags[position:]:
                    self.ignored_depth -= removed in IGNORED_TAGS
                    self.important_depth -= removed in IMPORTANT_TAGS
                del self.tags[position:]
                break

    def handle_data(self, data):
        if not self.ignored_depth:
            text = data.strip()
            if text:
                self.visible.append(text)
                if self.important_depth:
                    self.important.append(text)


def parse_html(content, *, on_link=None):
    # Plain text needs no HTML tree. Preserve its normal surrounding trim.
    if "<" not in content and "&" not in content:
        return content.strip(), ""
    if len(content) >= LARGE_HTML_CHARS:
        parser = LargePageText(on_link)
        parser.feed(content)
        parser.close()
        return " ".join(parser.visible), " ".join(parser.important)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(content, "html.parser")

    # remove irrelevant tags that may contain non-visible text or scripts
    for tag in soup(IGNORED_TAGS):
        tag.decompose()

    if on_link is not None:
        for tag in soup.find_all(LINK_TAGS, href=True):
            on_link(tag.name, tag["href"])

    all_text_parts = []
    important_text_parts = []

    # Iterate through all text nodes in the HTML document.
    for text_node in soup.find_all(string=True):

        # Skip document metadata that the parser represents as strings.
        if isinstance(text_node, (Comment, Declaration, Doctype, ProcessingInstruction)):
            continue

        text = str(text_node).strip()

        #skip empty text nodes
        if not text:
            continue

        all_text_parts.append(text)

        parent = text_node.parent
        is_important = False

        #check if the text node is inside an important tag
        while parent is not None:

            if parent.name in IMPORTANT_TAGS:
                is_important = True
                break

            parent = parent.parent

        if is_important:
            important_text_parts.append(text)

    return (
        " ".join(all_text_parts),
        " ".join(important_text_parts)
    )
