from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import random
import numpy as np
from math import hypot
from scipy.spatial import cKDTree

@dataclass
class GSPHConfig:
    time_budget_s: float = 1.5
    cand_m: int = 12
    cand_knn: int = 16
    use_delaunay: bool = False
    max_sweeps_2opt: int = 1
    or_opt_max_l: int = 1
    or_opt_move_limit: int = 3000
    ils_iters: int = 1
    skip_prob_2opt: float = 0.35
    eps_frontier: float = 5.0
    seed: Optional[int] = 42  

def euclidean_distance(p1: Tuple[float,float], p2: Tuple[float,float]) -> int:
    """Distancia euclidiana (entera TSPLIB-style)."""
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return int(dist + 0.5)

def eucl_float(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Distancia float para decisiones heurísticas."""
    return hypot(a[0]-b[0], a[1]-b[1])

def total_path_length(path: List[Tuple[float,float]]) -> int:
    if len(path) < 2: return 0
    return sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))

def total_tour_length(tour: List[Tuple[float,float]]) -> int:
    if len(tour) < 2: return 0
    return total_path_length(tour) + euclidean_distance(tour[-1], tour[0])
def subdivide_quadrants(pts: List[Tuple[float,float]]):
    xs, ys = zip(*pts)
    xmid, ymid = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    quads = {'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]}
    for p in pts:
        x, y = p
        if x <= xmid and y >  ymid: quads['Q1'].append(p)
        elif x >  xmid and y >  ymid: quads['Q2'].append(p)
        elif x <= xmid and y <= ymid: quads['Q3'].append(p)
        else: quads['Q4'].append(p)
    return quads, xmid, ymid

def best_frontier_pair(A, B, direction: str, mid: float, eps: float):
    filt = (lambda p: abs(p[0]-mid) < eps) if direction=='vertical' else (lambda p: abs(p[1]-mid) < eps)
    candA = [p for p in A if filt(p)]
    candB = [p for p in B if filt(p)]
    best, best_d = (None,None), float('inf')
    for a in candA:
        for b in candB:
            d = euclidean_distance(a,b)
            if d < best_d:
                best, best_d = (a,b), d
    return best, best_d
def tsp_2opt(points: List[Tuple[float,float]], max_iter: int = 50) -> List[Tuple[float,float]]:
    n = len(points)
    if n < 3:
        return points[:]
    route = points[:]
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                old = (euclidean_distance(route[i-1], route[i]) +
                       euclidean_distance(route[k],   route[k+1]))
                new = (euclidean_distance(route[i-1], route[k]) +
                       euclidean_distance(route[i],   route[k+1]))
                if new < old:
                    route[i:k+1] = reversed(route[i:k+1])
                    improved = True
        it += 1
    return route
def build_distance_matrix(points: List[Tuple[float,float]]) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff**2).sum(-1))
    D = np.asarray(D + 0.5, dtype=np.int32)
    np.fill_diagonal(D, 0)
    return D

def build_candidate_sets(points: List[Tuple[float,float]], m: int, knn_k: int, use_delaunay: bool) -> List[List[int]]:
    n = len(points)
    P = np.asarray(points, dtype=float)
    cand_sets = [set() for _ in range(n)]
    k = min(knn_k + 1, n)
    tree = cKDTree(P)
    _, idxs = tree.query(P, k=k)
    for i in range(n):
        neigh = idxs[i].tolist() if np.ndim(idxs[i]) else [int(idxs[i])]
        for j in neigh:
            if j != i:
                cand_sets[i].add(int(j))

    if use_delaunay:
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(P)
            for simplex in tri.simplices:
                a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
                cand_sets[a].update([b, c]); cand_sets[b].update([a, c]); cand_sets[c].update([a, b])
        except Exception:
            pass

    def _dsq(i, j):
        dx = P[i,0]-P[j,0]; dy = P[i,1]-P[j,1]
        return dx*dx + dy*dy

    out = []
    for i in range(n):
        arr = list(cand_sets[i])
        arr.sort(key=lambda j: _dsq(i, j))
        out.append(arr[:m])
    return out

def route_to_perm(route: List[Tuple[float,float]], nodes: List[Tuple[float,float]]) -> List[int]:
    idx = {tuple(nodes[i]): i for i in range(len(nodes))}
    return [idx[tuple(p)] for p in route]

def perm_to_route(perm: List[int], nodes: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    return [nodes[i] for i in perm]

def tour_len_from_perm(perm: List[int], D: np.ndarray) -> int:
    n = len(perm)
    if n < 2: return 0
    s = 0
    for i in range(n-1):
        s += D[perm[i], perm[i+1]]
    s += D[perm[-1], perm[0]]
    return int(s)

def ensure_valid_perm(perm: List[int], N: int, D: np.ndarray) -> List[int]:
    seen = set()
    perm2 = []
    for v in perm:
        if 0 <= v < N and v not in seen:
            perm2.append(v); seen.add(v)
    missing = [i for i in range(N) if i not in seen]
    if not perm2:
        return list(range(N))
    for m in missing:
        best_pos = 0
        best_inc = float("inf")
        L = len(perm2)
        for j in range(L):
            a = perm2[j]; b = perm2[(j+1) % L]
            inc = D[a, m] + D[m, b] - D[a, b]
            if inc < best_inc:
                best_inc = inc; best_pos = j + 1
        perm2.insert(best_pos, m)
    return perm2

def two_opt_dlb(perm: List[int], D: np.ndarray, cand: List[List[int]],
                max_sweeps: int, skip_prob: float, deadline: Optional[float]) -> List[int]:
    N = D.shape[0]
    n = len(perm)
    pos = [-1]*N
    for i, v in enumerate(perm):
        pos[v] = i
    dontlook = [False]*N

    for _ in range(max_sweeps):
        improved = True
        while improved:
            improved = False
            for a in range(n):
                if deadline and time.monotonic() > deadline:
                    return perm
                a1 = perm[a]
                if dontlook[a1]:
                    continue
                if skip_prob and random.random() < skip_prob:
                    continue
                b1 = perm[(a+1) % n]
                best_gain = 0
                best_c = None
                for cnode in cand[a1]:
                    c = pos[cnode]
                    if c == -1 or c == a or c == (a+1) % n:
                        continue
                    a2 = perm[c]; b2 = perm[(c+1) % n]
                    gain = (D[a1,b1] + D[a2,b2]) - (D[a1,a2] + D[b1,b2])
                    if gain > best_gain:
                        best_gain = gain; best_c = c
                if best_gain > 0 and best_c is not None:
                    i, k = a+1, best_c
                    if k < i:
                        i, k = k+1, a
                    perm[i:k+1] = reversed(perm[i:k+1])
                    for idx in range(i, k+1):
                        pos[perm[idx]] = idx
                    dontlook[a1] = False
                    improved = True
                else:
                    dontlook[a1] = True
    return perm

def or_opt_move(perm: List[int], D: np.ndarray, cand: List[List[int]],
                Lmax: int, move_limit: int, deadline: Optional[float]) -> List[int]:
    if Lmax <= 0: return perm
    N = D.shape[0]
    n = len(perm)
    if n < 6: return perm

    pos = [-1]*N
    for i, v in enumerate(perm):
        pos[v] = i

    moves = 0
    improved = True
    while improved and moves < move_limit:
        improved = False
        for i in range(n):
            if deadline and time.monotonic() > deadline:
                return perm
            for L in range(1, Lmax+1):
                end = (i + L - 1) % n
                anchors = cand[perm[i]]
                best_gain = 0
                best_j = None
                for jnode in anchors:
                    j = pos[jnode]
                    if j == -1: continue
                    inside = False
                    p = i
                    for _ in range(L):
                        if j == p: inside = True; break
                        p = (p + 1) % n
                    if inside or j == (i-1) % n or j == end:
                        continue
                    a, b = perm[(i-1)%n], perm[i]
                    c, d = perm[end],    perm[(end+1)%n]
                    e, f = perm[j],      perm[(j+1)%n]
                    removed = D[a,b] + D[c,d] + D[e,f]
                    added   = D[a,d] + D[e,b] + D[c,f]
                    gain = removed - added
                    if gain > best_gain:
                        best_gain = gain; best_j = j
                if best_gain > 0 and best_j is not None:
                    if end < i:
                        perm = perm[i:] + perm[:i]
                        best_j = (best_j - i) % n
                        i = 0; end = L-1
                        n = len(perm)
                    block = perm[i:i+L]
                    del perm[i:i+L]
                    if best_j > i:
                        best_j -= L
                    insert_at = best_j + 1
                    perm[insert_at:insert_at] = block
                    n = len(perm)
                    pos = [-1]*N
                    for ii, v in enumerate(perm):
                        pos[v] = ii
                    moves += 1
                    improved = True
                    break
            if improved or moves >= move_limit:
                break
    return perm

def double_bridge_on_perm(perm: List[int]) -> List[int]:
    n = len(perm)
    if n < 8: return perm[:]
    a = 1 + random.randint(0, n//4 - 1)
    b = a + random.randint(1, n//4)
    c = b + random.randint(1, n//4)
    d = c + random.randint(1, n//4)
    A = perm[:a]; B = perm[a:b]; C = perm[b:c]; Dd = perm[c:d]; E = perm[d:]
    return A + Dd + C + B + E

def polish_fast(nodes: List[Tuple[float,float]], route: List[Tuple[float,float]], cfg: GSPHConfig) -> List[Tuple[float,float]]:
    if len(route) < 10:
        return route[:]
    D = build_distance_matrix(nodes)
    cand = build_candidate_sets(nodes, m=cfg.cand_m, knn_k=cfg.cand_knn, use_delaunay=cfg.use_delaunay)

    deadline = time.monotonic() + max(0.05, cfg.time_budget_s)
    raw_perm = route_to_perm(route, nodes)
    perm = ensure_valid_perm(raw_perm, len(nodes), D)
    perm = two_opt_dlb(perm, D, cand, max_sweeps=cfg.max_sweeps_2opt, skip_prob=cfg.skip_prob_2opt, deadline=deadline)
    perm = or_opt_move(perm, D, cand, Lmax=cfg.or_opt_max_l, move_limit=cfg.or_opt_move_limit, deadline=deadline)
    perm = two_opt_dlb(perm, D, cand, max_sweeps=1, skip_prob=cfg.skip_prob_2opt, deadline=deadline)

    best = perm[:]; best_len = tour_len_from_perm(best, D)
    for _ in range(max(0, cfg.ils_iters)):
        if time.monotonic() > deadline: break
        kick = double_bridge_on_perm(best)
        kick = ensure_valid_perm(kick, len(nodes), D)
        kick = two_opt_dlb(kick, D, cand, max_sweeps=1, skip_prob=cfg.skip_prob_2opt, deadline=deadline)
        L = tour_len_from_perm(kick, D)
        if L < best_len:
            best, best_len = kick[:], L

    return perm_to_route(best, nodes)

def _concat_routes_q_order(routes: Dict[str, List[Tuple[float,float]]]) -> List[Tuple[float,float]]:
    order = ['Q1','Q2','Q4','Q3']
    seq: List[Tuple[float,float]] = []
    last = None
    for key in order:
        r = routes.get(key, [])
        if not r: 
            continue
        if last is None:
            seq = r[:]
        else:
            d1 = eucl_float(last, r[0]) if r else 1e9
            d2 = eucl_float(last, r[-1]) if r else 1e9
            seq += (r if d1 <= d2 else list(reversed(r)))
        last = seq[-1] if seq else None
    return seq
def gsph_fc(nodes: List[Tuple[float,float]], cfg: GSPHConfig) -> Tuple[Dict[str, List[Tuple[float,float]]],
                                                                      List[Tuple[Tuple[float,float], Tuple[float,float]]],
                                                                      int, float, float]:
    """
    nodes: lista de (x,y) (metros). Devuelve:
      - routes_out: {'QALL': tour_mejorado, ...}
      - connections: lista de pares (a_xy, b_xy) conectando cuadrantes
      - total_len: longitud entera TSPLIB del tour final
      - xmid, ymid: fronteras de cuadrantes
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    
    quads, xmid, ymid = subdivide_quadrants(nodes)

    routes: Dict[str, List[Tuple[float,float]]] = {}
    for q, pts in quads.items():
        if len(pts) > 1:
            start = pts[0]
            remaining = set(pts); remaining.remove(start)
            curr = start; seed_path = [curr]
            while remaining:
                nxt = min(remaining, key=lambda p: eucl_float(curr, p))
                seed_path.append(nxt); remaining.remove(nxt); curr = nxt
            rt = seed_path
            for _ in range(2):
                rt = tsp_2opt(rt, max_iter=50)
            routes[q] = rt
        else:
            routes[q] = pts

    neighbor_pairs = [
        ('Q1','Q2','vertical',  xmid),
        ('Q1','Q3','horizontal',ymid),
        ('Q2','Q4','horizontal',ymid),
        ('Q3','Q4','vertical',  xmid)
    ]
    connections: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    for q1,q2,dirc,mid in neighbor_pairs:
        (a,b), d = best_frontier_pair(quads[q1], quads[q2], dirc, mid, cfg.eps_frontier)
        if a and b:
            connections.append((a,b))
    initial_tour = _concat_routes_q_order(routes)
    improved_tour = polish_fast(nodes, initial_tour, cfg)
    total_len = total_tour_length(improved_tour)

    routes_out = {'QALL': improved_tour}
    return routes_out, connections, total_len, xmid, ymid

def solve_xy(nodes_xy: List[Tuple[float,float]], config: Optional[GSPHConfig] = None) -> Dict[str, Any]:
    """
    Ejecuta GSPH–FC sobre XY (metros) y devuelve artefactos clave.
    nodes_xy: lista (incluye depósito y órdenes si así lo defines afuera,
              o solo órdenes si manejas depósito en proyección externa).
    """
    cfg = config or GSPHConfig()
    routes_out, connections, total_len, xmid, ymid = gsph_fc(nodes_xy, cfg)

    tour_xy = routes_out.get("QALL", [])
    dist_m = 0.0
    if tour_xy:
        for i in range(len(tour_xy)-1):
            a, b = tour_xy[i], tour_xy[i+1]
            dist_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        a, b = tour_xy[-1], tour_xy[0]
        dist_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    return {
        "route_xy": tour_xy,
        "connections_xy": connections,
        "metrics": {
            "total_distance_km": round(dist_m / 1000.0, 3),
            "total_len_integer": int(total_len),
            "nodes": len(nodes_xy)
        },
        "quadrants": {"xmid": xmid, "ymid": ymid}
    }
