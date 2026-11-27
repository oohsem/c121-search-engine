"""Build and rank the sparse link graph offline."""
import json
import struct
from array import array
from tempfile import TemporaryFile
from urllib.parse import urljoin, urlsplit, urlunsplit

from compact_format import PAGERANK_SCORE


EDGE = struct.Struct("<II")  # source URL ID, destination URL ID
EDGE_BLOCK_BYTES = 64 * 1024


def canonical_url(url):
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if (parts.scheme not in ("http", "https") or not host
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

    def add(self, tag, href):
        href = href.strip()
        if tag == "base":
            if self.base is None:
                self.base = resolve_link(self.url, href) if href else self.url
        elif href:
            self.hrefs.add(href)

    def targets(self):
        base = self.base or self.url
        return sorted({target for href in self.hrefs if (target := resolve_link(base, href))})


def solve(degrees, edges, damping, iterations, tol):
    n = len(degrees)
    if not n:
        return array("d")

    # The lecture initializes every rank to one. Uniform teleportation and
    # dangling-page redistribution keep the mean rank at one.
    rank = array("d", [1.0]) * n
    for _ in range(iterations):
        dangling = sum(value for value, degree in zip(rank, degrees) if not degree)
        new = array("d", [1.0 - damping + damping * dangling / n]) * n
        shares = array("d", (damping * value / degree if degree else 0.0
                             for value, degree in zip(rank, degrees)))
        for source, target in edges():
            new[target] += shares[source]
        delta = max(abs(a - b) for a, b in zip(new, rank))
        rank = new
        if delta < tol:
            break
    return rank


def pagerank(n, out_edges, damping=0.85, iterations=50, tol=1e-9):
    """Rank an in-memory adjacency list, useful for small graphs."""
    rows = list(out_edges)
    if len(rows) != n:
        raise ValueError("Expected one adjacency row per URL.")

    def edges():
        return ((source, target) for source, targets in enumerate(rows) for target in targets)

    return list(solve([len(row) for row in rows], edges, damping, iterations, tol))


def iter_edges(graph):
    graph.seek(0)
    while block := graph.read(EDGE_BLOCK_BYTES):
        yield from EDGE.iter_unpack(block)


def write_pagerank(documents_path, links_path, scores_path):
    """Rank only indexed URLs and store one score for each indexed document."""
    node_ids = {}
    link_rows = []
    document_nodes = array("I")

    with documents_path.open(encoding="utf-8") as documents, links_path.open("rb") as links:
        for line in documents:
            record = json.loads(line)
            if record["id"] != len(document_nodes):
                raise ValueError("Nonconsecutive document IDs in the link graph.")
            url = canonical_url(record["url"]) or record["url"]
            node = node_ids.get(url)
            if node is None:
                node = node_ids[url] = len(node_ids)
                link_rows.append([])
            document_nodes.append(node)
            link_rows[node].append(links.tell())
            if not links.readline():
                raise ValueError("Missing document link row.")
        if links.read(1):
            raise ValueError("Extra document link rows.")

        # Union the neighborhoods of duplicate URLs. Only one neighborhood
        # is in memory; the complete directed edge list stays on disk.
        with TemporaryFile(dir=links_path.parent) as graph:
            degrees = array("I")
            for source, offsets in enumerate(link_rows):
                targets = set()
                for offset in offsets:
                    links.seek(offset)
                    for url in json.loads(links.readline()):
                        target = node_ids.get(url)
                        if target is not None and target != source:
                            targets.add(target)
                degrees.append(len(targets))
                for target in sorted(targets):
                    graph.write(EDGE.pack(source, target))

            ranks = solve(degrees, lambda: iter_edges(graph), 0.85, 50, 1e-9)

    with scores_path.open("wb") as scores:
        for node in document_nodes:
            scores.write(PAGERANK_SCORE.pack(ranks[node]))
    print(f"PageRank computed on {len(degrees):,} URLs and {sum(degrees):,} directed links.")
