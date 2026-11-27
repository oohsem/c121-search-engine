"""Resolve indexed URLs and collect graph links and anchor labels."""
from urllib.parse import urljoin, urlsplit, urlunsplit


def canonical_url(url):
    url = url.strip().split("#", 1)[0]
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if (parts.scheme not in ("http", "https") or not host or port == 0
            or parts.username is not None or any(c.isspace() or not c.isprintable() for c in host)):
        return None
    host = f"[{host}]" if ":" in host else host
    default_port = 80 if parts.scheme == "http" else 443
    authority = host if port in (None, default_port) else f"{host}:{port}"
    normalized = urlunsplit((parts.scheme, authority, parts.path or "/", parts.query, ""))
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return normalized


def resolve_link(base, href):
    if not href or any(ord(c) < 32 or ord(c) == 127 for c in href):
        return None
    try:
        return canonical_url(urljoin(base, href))
    except ValueError:
        return None


class PageLinks:
    def __init__(self, url):
        self.url = url
        self.base = None
        self.hrefs = set()
        self.anchor_labels = {}

    def add(self, tag, href):
        href = href.strip()
        if tag == "base":
            if self.base is None:
                self.base = resolve_link(self.url, href) if href else self.url
        elif href:
            self.hrefs.add(href)

    def targets(self):
        return self.rows()[0]

    def add_anchor(self, href, label):
        href = href.strip()
        if href and label:
            self.anchor_labels.setdefault(href, set()).add(label)

    def anchored_targets(self):
        return iter(self.rows()[1])

    def rows(self):
        """Resolve each raw href once for both indexing artifacts."""
        base = self.base or self.url
        targets = set()
        anchored = {}
        for href in self.hrefs | self.anchor_labels.keys():
            target = resolve_link(base, href)
            if target is None:
                continue
            if href in self.hrefs:
                targets.add(target)
            labels = self.anchor_labels.get(href)
            if labels:
                anchored.setdefault(target, set()).update(labels)
        return sorted(targets), [(target, sorted(labels)) for target, labels in sorted(anchored.items())]
