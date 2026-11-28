"""Query-dependent HITS scores on the indexed link neighborhood."""
import sqlite3
from contextlib import closing
from itertools import islice, zip_longest


HITS_GRAPH = "hits.sqlite3"
HITS_ROOT_SIZE = 40
MAX_NEIGHBORS = 20
MAX_BASE_SIZE = 300
MAX_ITERATIONS = 10
TOLERANCE = 1e-6


def write_hits_graph(path, document_nodes, degrees, edges):
    """Save the PageRank graph with lookups in both directions."""
    with closing(sqlite3.connect(path)) as database:
        database.executescript("""
            CREATE TABLE documents (id INTEGER PRIMARY KEY, node INTEGER NOT NULL);
            CREATE TABLE nodes (id INTEGER PRIMARY KEY, degree INTEGER NOT NULL);
            CREATE TABLE edges (
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                PRIMARY KEY(source, target)
            ) WITHOUT ROWID;
        """)
        with database:
            database.executemany("INSERT INTO documents VALUES (?, ?)", enumerate(document_nodes))
            database.executemany("INSERT INTO nodes VALUES (?, ?)", enumerate(degrees))
            database.executemany("INSERT INTO edges VALUES (?, ?)", edges)
            database.execute("CREATE INDEX incoming ON edges (target, source)")


def normalize_scores(values):
    length = sum(values)
    return [value / length for value in values] if length else values


def hits(count, edges):
    """Update authorities from hubs, then hubs from authorities."""
    edges = tuple(edges)
    authorities = hubs = [1.0] * count
    for _ in range(MAX_ITERATIONS):
        next_authorities = [0.0] * count
        for source, target in edges:
            next_authorities[target] += hubs[source]
        next_authorities = normalize_scores(next_authorities)

        next_hubs = [0.0] * count
        for source, target in edges:
            next_hubs[source] += next_authorities[target]
        next_hubs = normalize_scores(next_hubs)
        delta = max((abs(a - b) for a, b in zip(hubs, next_hubs)), default=0.0)
        authorities, hubs = next_authorities, next_hubs
        if delta < TOLERANCE:
            break
    return hubs, authorities


class LinkGraph:
    def __init__(self, database):
        self.database = database

    @classmethod
    def open(cls, path):
        if not path.exists():
            return None
        database = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        try:
            database.execute("SELECT id, node FROM documents LIMIT 0")
            database.execute("SELECT id, degree FROM nodes LIMIT 0")
            database.execute("SELECT source, target FROM edges INDEXED BY incoming LIMIT 0")
        except sqlite3.Error:
            database.close()
            raise
        return cls(database)

    def close(self):
        self.database.close()

    def scores(self, documents):
        documents = tuple(dict.fromkeys(islice(documents, HITS_ROOT_SIZE)))
        if not documents:
            return {}
        placeholders = ",".join("?" for _ in documents)
        nodes = dict(self.database.execute(
            f"SELECT id, node FROM documents WHERE id IN ({placeholders})", documents,
        ))
        if len(nodes) != len(documents):
            raise ValueError("missing graph node for a ranked document")
        roots = dict.fromkeys(nodes[doc] for doc in documents)

        base = self._expand(roots)
        indices = {node: index for index, node in enumerate(sorted(base))}
        edges = self._induced_edges(indices)
        if not edges:
            return {}
        hubs, authorities = hits(len(indices), edges)
        return {doc: (hubs[indices[node]], authorities[indices[node]]) for doc, node in nodes.items()}

    def _expand(self, roots):
        base = set(roots)
        if not base:
            return base
        placeholders = ",".join("?" for _ in roots)
        queries = (
            f"SELECT target FROM edges WHERE source = ? AND target NOT IN ({placeholders}) "
            "ORDER BY target LIMIT ?",
            f"SELECT source FROM edges INDEXED BY incoming WHERE target = ? AND source NOT IN ({placeholders}) "
            "ORDER BY source LIMIT ?",
        )
        neighborhoods = []
        for root in roots:
            for query in queries:
                neighborhoods.append(self.database.execute(query, (root, *roots, MAX_NEIGHBORS)).fetchall())
        for neighbors in zip_longest(*neighborhoods):
            for row in neighbors:
                if row is not None:
                    base.add(row[0])
                if len(base) == MAX_BASE_SIZE:
                    return base
        return base

    def _induced_edges(self, indices):
        placeholders = ",".join("?" for _ in indices)
        degrees = dict(self.database.execute(
            f"SELECT id, degree FROM nodes WHERE id IN ({placeholders})", tuple(indices),
        ))
        if len(degrees) != len(indices):
            raise ValueError("missing node in the HITS neighborhood")

        regular = tuple(node for node, degree in degrees.items() if degree <= len(indices))
        edges = []
        if regular:
            regular_slots = ",".join("?" for _ in regular)
            pairs = self.database.execute(
                f"SELECT source, target FROM edges WHERE source IN ({regular_slots})", regular,
            )
            edges.extend((indices[source], indices[target]) for source, target in pairs if target in indices)

        seek = f"SELECT target FROM edges WHERE source = ? AND target IN ({placeholders})"
        for node, degree in degrees.items():
            if degree > len(indices):
                targets = self.database.execute(seek, (node, *indices))
                edges.extend((indices[node], indices[target]) for target, in targets)
        return edges
