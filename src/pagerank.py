
def pagerank(n, out_edges, damping=0.85, iterations=50, tol=1e-9):
    # Lecture 24, p52: PR(A) = (1-d) + d * sum(PR(Ti) / C(Ti))
    #   d     = damping/teleport factor (usually 0.85)
    #   C(Ti) = number of outlinks of Ti, summed over every page Ti linking to A
    # n: number of docs. out_edges[i]: list of doc IDs that doc i links to.
    # Ranks average 1 (they sum to n), not 1 total (p56).
    if n == 0:
        return []

    # p53: "PR(p) = 1 for starters", then iterate until it converges (p56).
    rank = [1.0] * n

    for _ in range(iterations):
        new = [1.0 - damping] * n
        dangling = 0.0

        for i, targets in enumerate(out_edges):
            if targets:
                share = damping * rank[i] / len(targets)  # d * PR(Ti) / C(Ti)
                for t in targets:
                    new[t] += share
            else:
                # p37-38: at a dead end, jump to a random page.
                dangling += rank[i]

        spread = damping * dangling / n
        new = [r + spread for r in new]

        delta = max(abs(a - b) for a, b in zip(new, rank))
        rank = new
        if delta < tol:
            break

    return rank
