"""Build short local previews of ranked documents on request."""
import json
import re
from urllib.parse import urldefrag

from parser import parse_html
from tokenizer import stem_word


WORDS = re.compile(r"[^\W_]+")
MAX_SOURCE_BYTES = 512_000
PREVIEW_CHARS = 160


def matching_excerpt(terms, dataset, relative_path, indexed_url):
    if not relative_path:
        return ""
    try:
        page = (dataset / relative_path).resolve()
        if not page.is_relative_to(dataset) or page.stat().st_size > MAX_SOURCE_BYTES:
            return ""
        record = json.loads(page.read_text(encoding="utf-8"))
        source_url = record["url"]
        if not isinstance(source_url, str) or urldefrag(source_url.strip()).url != indexed_url:
            return ""
        visible, _ = parse_html(record["content"])
    except (OSError, ValueError, TypeError, KeyError):
        return ""
    text = " ".join(visible.split())
    for match in WORDS.finditer(text):
        if stem_word(match.group().lower()) in terms:
            start = max(0, match.start() - 45)
            end = start + PREVIEW_CHARS
            excerpt = ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")
            return "".join(character for character in excerpt if character.isprintable())
    return ""
