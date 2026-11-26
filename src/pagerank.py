 
def pagerank(n, out_edges, damping=0.85, iterations=50, tol=1e-9):
    # n: number of docs. out_edges[i]: list of doc IDs that doc i links to.
    # Power iteration. Dangling pages spread their rank evenly to all pages.
    if n == 0:
        return []
    rank = [1.0 / n] * n
    for _ in range(iterations):
        new = [(1.0 - damping) / n] * n
        dangling = 0.0
        for i, targets in enumerate(out_edges):
            if targets:
                share = damping * rank[i] / len(targets)
                for t in targets:
                    new[t] += share
            else:
                dangling += rank[i]
        spread = damping * dangling / n
        new = [r + spread for r in new]
        delta = sum(abs(a - b) for a, b in zip(new, rank))
        rank = new
        if delta < tol:
            break
    return rank
