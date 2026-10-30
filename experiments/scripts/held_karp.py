"""
Óptimo exacto por programación dinámica de Held-Karp (O(n^2 2^n)) para instancias
chicas. Se usa para certificar los óptimos de eil10 y eil20 (subconjuntos de eil51
sin BKS publicada): 159 y 243.

Uso:  python experiments/scripts/held_karp.py eil10 eil20
"""
import os, sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "algorithms"))
import GSPH_V2 as g  # noqa: E402
from GSPH_V2 import njit  # noqa: E402

STANDARD_DIR = os.path.join(SCRIPT_DIR, "..", "instances", "standard")


@njit(cache=True)
def held_karp(D):
    """Largo del ciclo óptimo; el nodo 0 es el origen y las máscaras cubren los nodos 1..n-1."""
    n = D.shape[0]
    INF = 1 << 60
    dp = np.full((1 << (n - 1), n - 1), INF, np.int64)
    for j in range(n - 1):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, 1 << (n - 1)):
        for j in range(n - 1):
            v = dp[mask, j]
            if v >= INF or not (mask >> j) & 1:
                continue
            for k in range(n - 1):
                if (mask >> k) & 1:
                    continue
                c = v + D[j + 1, k + 1]
                nm = mask | (1 << k)
                if c < dp[nm, k]:
                    dp[nm, k] = c
    full = (1 << (n - 1)) - 1
    best = INF
    for j in range(n - 1):
        c = dp[full, j] + D[j + 1, 0]
        if c < best:
            best = c
    return best


def main():
    nombres = sys.argv[1:] or ["eil10", "eil20"]
    for name in nombres:
        pts = g.read_tsplib(os.path.join(STANDARD_DIR, f"{name}.tsp"))
        if len(pts) > 23:
            sys.exit(f"{name}: n={len(pts)} es demasiado grande para Held-Karp en memoria")
        D = g.build_distance_matrix(pts).astype(np.int64)
        print(f"{name} (n={len(pts)}): óptimo = {held_karp(D)}")


if __name__ == "__main__":
    main()
