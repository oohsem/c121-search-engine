from bs4 import (BeautifulSoup, Comment, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning)

import warnings


# Ignore warnings when XML or URL-like content is encountered.
warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning
)

warnings.filterwarnings(
    "ignore",
    category=MarkupResemblesLocatorWarning
)



IMPORTANT_TAGS = {"title", "h1", "h2", "h3", "b", "strong"}

IGNORED_TAGS = {"script", "style", "noscript", "template"}


def parse_html(content):
    #Extract all visible text and important text from HTML.


    soup = BeautifulSoup(content, "html.parser")

    # remove irrelevant tags that may contain non-visible text or scripts
    for tag in soup(IGNORED_TAGS):
        tag.decompose()

    all_text_parts = []
    important_text_parts = []

    # Iterate through all text nodes in the HTML document.
    for text_node in soup.find_all(string=True):

        # Skip comments and empty text nodes.
        if isinstance(text_node, Comment):
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
