import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from bs4.element import Comment, Declaration, Doctype, ProcessingInstruction


IMPORTANT_TAGS = {"title", "h1", "h2", "h3", "b", "strong"}

IGNORED_TAGS = {"script", "style", "noscript", "template"}


def parse_html(content):
    #Extract all visible text and important text from HTML.


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(content, "html.parser")

    # remove irrelevant tags that may contain non-visible text or scripts
    for tag in soup(IGNORED_TAGS):
        tag.decompose()

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
