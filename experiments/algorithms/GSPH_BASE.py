import os
import time
import random
import numpy as np
import sys
from math import hypot
from itertools import product
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None
try:
    import psutil
except Exception:
    psutil = None
try:
    import tracemalloc
except Exception:
    tracemalloc = None
try:
    from pympler import tracker
except Exception:
    tracker = None
try:
    from memory_profiler import memory_usage
except Exception:
    memory_usage = None

EPS_FRONTIER      = 5
MAX_ITER_LOCAL    = 800
RESULTS_DIR       = "../results"
BASE_DIR          = "../instances/standard"

TIME_BUDGET_S     = 5.0   # Más tiempo para optimización (1.5 -> 5.0)
CAND_M            = 20    # Más candidatos por nodo (12 -> 20) 
CAND_KNN          = 25    # Más vecinos cercanos (16 -> 25)
USE_DELAUNAY      = True  # Activar Delaunay para mejor conectividad
MAX_SWEEPS_2OPT   = 3     # Más pasadas de 2-opt (1 -> 3)
OR_OPT_MAX_L      = 3     # Cadenas más largas Or-opt (1 -> 3)
OR_OPT_MOVE_LIMIT = 8000  # Más movimientos Or-opt (3000 -> 8000)
ILS_ITERS         = 5     # Más iteraciones ILS (1 -> 5)
SKIP_PROB_2OPT    = 0.15  # Menos probabilidad de saltar (0.35 -> 0.15) 

FRONTIER_TOPK               = 3
MAX_ASSEMBLY_PER_PARTITION  = 8
ASSEMBLY_FINALISTS          = 3
PARTITION_PORTFOLIO_LIMIT   = 6
PARTITION_SHIFT_FRAC        = 0.12
LIGHT_POLISH_FRACTION       = 0.20
LIGHT_MAX_SWEEPS_2OPT       = 1
LIGHT_OR_OPT_L              = 1

def _env_int(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except Exception:
        return default

PROFILE_RUNS = _env_int('GSPH_PROFILE_RUNS', 10)

Q_ORDER = ['Q1', 'Q2', 'Q4', 'Q3']
ADJACENCY_CYCLE = [
    ('Q1', 'Q2', 'vertical'),
    ('Q2', 'Q4', 'horizontal'),
    ('Q4', 'Q3', 'vertical'),
    ('Q3', 'Q1', 'horizontal'),
]

_event_enabled = False
_events = []
_events_t0 = None
def event_enable(t0=None):
    global _event_enabled, _events, _events_t0
    _events = []
    _events_t0 = t0 if t0 is not None else time.time()
    _event_enabled = True
def event_disable():
    global _event_enabled
    _event_enabled = False
def record_event(name):
    try:
        if not _event_enabled:
            return
        _events.append((time.time() - _events_t0, name))
    except Exception:
        pass


def read_tsplib(filename):
    nodes = []
    with open(filename, 'r') as f:
        reading_nodes = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if line == "EOF":
                break
            if reading_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    nodes.append((x, y))
    return nodes

def euclidean_distance(p1, p2):
    """TSPLIB EUC_2D redondeada (entera)."""
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return int(dist + 0.5)

def eucl_float(a, b):
    """Distancia float (sin redondeo) para decisiones heurísticas rápidas."""
    return hypot(a[0]-b[0], a[1]-b[1])

def total_path_length(path):
    if len(path) < 2: return 0
    return sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))

def total_tour_length(tour):
    if len(tour) < 2: return 0
    return total_path_length(tour) + euclidean_distance(tour[-1], tour[0])

def subdivide_quadrants(pts, xmid=None, ymid=None):
    if not pts:
        return {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}, 0.0, 0.0
    xs, ys = zip(*pts)
    if xmid is None:
        xmid = (min(xs)+max(xs))/2
    if ymid is None:
        ymid = (min(ys)+max(ys))/2
    quads = {'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]}
    for p in pts:
        x, y = p
        if x <= xmid and y >  ymid: quads['Q1'].append(p)
        elif x >  xmid and y >  ymid: quads['Q2'].append(p)
        elif x <= xmid and y <= ymid: quads['Q3'].append(p)
        else: quads['Q4'].append(p)
    return quads, xmid, ymid

def generate_partition_portfolio(pts, shift_frac=PARTITION_SHIFT_FRAC, limit=PARTITION_PORTFOLIO_LIMIT):
    if not pts:
        return [('bbox', 0.0, 0.0)]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox_xmid = (xmin + xmax) / 2
    bbox_ymid = (ymin + ymax) / 2
    median_xmid = float(np.median(xs))
    median_ymid = float(np.median(ys))
    shift_x = (xmax - xmin) * shift_frac
    shift_y = (ymax - ymin) * shift_frac

    specs = [
        ('bbox', bbox_xmid, bbox_ymid),
        ('median', median_xmid, median_ymid),
        ('shift_ne', bbox_xmid + shift_x, bbox_ymid + shift_y),
        ('shift_nw', bbox_xmid - shift_x, bbox_ymid + shift_y),
        ('shift_se', bbox_xmid + shift_x, bbox_ymid - shift_y),
        ('shift_sw', bbox_xmid - shift_x, bbox_ymid - shift_y),
    ]

    unique = []
    seen = set()
    for label, xmid, ymid in specs:
        key = (round(xmid, 6), round(ymid, 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, xmid, ymid))
        if len(unique) >= limit:
            break
    return unique

def best_frontier_pairs(A, B, direction, mid, eps, k=FRONTIER_TOPK):
    if not A or not B:
        return [((None, None), float('inf'))]

    def _candidates(scale):
        band = eps * scale
        filt = (lambda p: abs(p[0]-mid) < band) if direction == 'vertical' else (lambda p: abs(p[1]-mid) < band)
        return [p for p in A if filt(p)], [p for p in B if filt(p)]

    candA, candB = _candidates(1)
    scale = 1
    while (not candA or not candB) and scale < 16:
        scale *= 2
        candA, candB = _candidates(scale)

    if not candA:
        candA = list(A)
    if not candB:
        candB = list(B)

    pairs = []
    for a in candA:
        for b in candB:
            d = euclidean_distance(a,b)
            pairs.append(((a, b), d))

    pairs.sort(key=lambda item: item[1])
    top = []
    seen = set()
    for item in pairs:
        key = item[0]
        if key in seen:
            continue
        seen.add(key)
        top.append(item)
        if len(top) >= max(1, k):
            break
    return top or [((None, None), float('inf'))]

def best_frontier_pair(A, B, direction, mid, eps):
    return best_frontier_pairs(A, B, direction, mid, eps, k=1)[0]

def build_anchored_path(points, start, end, max_iter=50):
    """
    Construye una ruta abierta con extremos fijados por las fronteras.
    El interior se completa con inserción de costo mínimo y luego se
    refina con 2-opt sin mover los extremos.
    """
    if not points:
        return []
    if len(points) == 1:
        return points[:]
    if start is None or end is None or start not in points or end not in points or start == end:
        start = points[0]
        remaining = set(points)
        remaining.remove(start)
        route = [start]
        curr = start
        while remaining:
            nxt = min(remaining, key=lambda p: eucl_float(curr, p))
            route.append(nxt)
            remaining.remove(nxt)
            curr = nxt
        return tsp_2opt(route, max_iter=max_iter)

    route = [start, end]
    remaining = [p for p in points if p != start and p != end]

    while remaining:
        best_node = None
        best_pos = 1
        best_inc = float("inf")
        for node in remaining:
            for i in range(len(route) - 1):
                inc = (euclidean_distance(route[i], node) +
                       euclidean_distance(node, route[i + 1]) -
                       euclidean_distance(route[i], route[i + 1]))
                if inc < best_inc:
                    best_inc = inc
                    best_node = node
                    best_pos = i + 1
        route.insert(best_pos, best_node)
        remaining.remove(best_node)

    return tsp_2opt(route, max_iter=max_iter)

def tsp_2opt(points, max_iter=50):
    """
    2-opt simple para PATH abierto usando euclidean_distance.
    Devuelve la ruta mejorada.
    """
    record_event('tsp_2opt_entry')
    n = len(points)
    if n < 3:
        record_event('tsp_2opt_exit')
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
    record_event('tsp_2opt_exit')
    return route
def build_distance_matrix(points):
    P = np.asarray(points, dtype=float)
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff**2).sum(-1))
    D = np.asarray(D + 0.5, dtype=np.int32)
    np.fill_diagonal(D, 0)
    return D
def build_candidate_sets(points, m=CAND_M, knn_k=CAND_KNN, use_delaunay=USE_DELAUNAY):
    """Solo kNN por defecto (rápido). Delaunay opcional."""
    n = len(points)
    P = np.asarray(points, dtype=float)
    cand_sets = [set() for _ in range(n)]
    k = min(knn_k + 1, n)
    if cKDTree is not None:
        tree = cKDTree(P)
        _, idxs = tree.query(P, k=k)
        for i in range(n):
            neigh = idxs[i].tolist() if np.ndim(idxs[i]) else [int(idxs[i])]
            for j in neigh:
                if j != i:
                    cand_sets[i].add(int(j))
    else:
        for i in range(n):
            order = sorted(range(n), key=lambda j: ((P[i, 0] - P[j, 0]) ** 2 + (P[i, 1] - P[j, 1]) ** 2))
            for j in order[1:k]:
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
    for i in range(n):
        for j in list(cand_sets[i]):
            cand_sets[j].add(i)
    def _dsq(i, j):
        dx = P[i,0]-P[j,0]; dy = P[i,1]-P[j,1]
        return dx*dx + dy*dy
    out = []
    for i in range(n):
        arr = list(cand_sets[i])
        arr.sort(key=lambda j: _dsq(i, j))
        out.append(arr[:m])
    return out

def route_to_perm(route, nodes):
    idx = {tuple(nodes[i]): i for i in range(len(nodes))}
    return [idx[tuple(p)] for p in route]

def perm_to_route(perm, nodes):
    return [nodes[i] for i in perm]

def tour_len_from_perm(perm, D):
    n = len(perm)
    if n < 2: return 0
    s = 0
    for i in range(n-1):
        s += D[perm[i], perm[i+1]]
    s += D[perm[-1], perm[0]]
    return int(s)
def ensure_valid_perm(perm, N, D):
    """
    Limpia 'perm': quita duplicados preservando orden y añade faltantes
    insertándolos en la mejor posición (criterio de inserción mínima).
    """
    seen = set()
    perm2 = []
    for v in perm:
        if 0 <= v < N and v not in seen:
            perm2.append(v)
            seen.add(v)
    missing = [i for i in range(N) if i not in seen]
    if not perm2:
        return list(range(N))
    for m in missing:
        best_pos = 0
        best_inc = float("inf")
        L = len(perm2)
        for j in range(L):
            a = perm2[j]
            b = perm2[(j+1) % L]
            inc = D[a, m] + D[m, b] - D[a, b]
            if inc < best_inc:
                best_inc = inc
                best_pos = j + 1
        perm2.insert(best_pos, m)
    return perm2
def two_opt_dlb(perm, D, cand, max_sweeps=1, skip_prob=SKIP_PROB_2OPT, deadline=None):
    record_event('two_opt_dlb_entry')
    N = D.shape[0]
    n = len(perm)
    pos = [-1]*N                 
    for i, v in enumerate(perm):
        pos[v] = i
    dontlook = [False]*N

    for _ in range(max_sweeps):
        record_event('two_opt_sweep_start')
        improved = True
        while improved:
            record_event('two_opt_iteration_start')
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
                        best_gain = gain
                        best_c = c
                if best_gain > 0 and best_c is not None:
                    i, k = a+1, best_c
                    if k < i:
                        i, k = k+1, a
                    perm[i:k+1] = reversed(perm[i:k+1])
                    for idx in range(i, k+1):
                        pos[perm[idx]] = idx
                    touched = {
                        a1,
                        b1,
                        perm[i],
                        perm[k],
                        perm[(i-1) % n],
                        perm[(k+1) % n],
                    }
                    for node in touched:
                        dontlook[node] = False
                    improved = True
                else:
                    dontlook[a1] = True
    record_event('two_opt_dlb_exit')
    return perm

def or_opt_move(perm, D, cand, Lmax=OR_OPT_MAX_L, move_limit=OR_OPT_MOVE_LIMIT, deadline=None):
    """Mueve bloques cortos en tour circular. L=1 por defecto para máxima velocidad."""
    record_event('or_opt_move_entry')
    if Lmax <= 0:
        return perm
    N = D.shape[0]
    n = len(perm)
    if n < 6:
        return perm
    pos = [-1]*N
    for i, v in enumerate(perm):
        pos[v] = i

    moves = 0
    improved = True
    while improved and moves < move_limit:
        record_event('or_opt_outer_loop')
        improved = False
        for i in range(n):
            if deadline and time.monotonic() > deadline:
                return perm
            for L in range(1, Lmax+1):
                end = (i + L - 1) % n
                anchors = list(set(cand[perm[i]]) | set(cand[perm[end]]))
                best_gain = 0
                best_j = None
                for jnode in anchors:
                    j = pos[jnode]
                    if j == -1:
                        continue
                    inside = False
                    p = i
                    for _ in range(L):
                        if j == p:
                            inside = True; break
                        p = (p + 1) % n
                    if inside or j == (i-1) % n or j == end:
                        continue
                    a, b = perm[(i-1)%n], perm[i]
                    c, d = perm[end], perm[(end+1)%n]
                    e, f = perm[j],  perm[(j+1)%n]
                    removed = D[a,b] + D[c,d] + D[e,f]
                    added   = D[a,d] + D[e,b] + D[c,f]
                    gain = removed - added
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
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
    record_event('or_opt_move_exit')
    return perm

def double_bridge_on_perm(perm):
    n = len(perm)
    if n < 8: return perm[:]
    a = 1 + random.randint(0, n//4 - 1)
    b = a + random.randint(1, n//4)
    c = b + random.randint(1, n//4)
    d = c + random.randint(1, n//4)
    A = perm[:a]; B = perm[a:b]; C = perm[b:c]; Dd = perm[c:d]; E = perm[d:]
    return A + Dd + C + B + E

def _polish_with_config(nodes, route, D, cand, budget_s, initial_sweeps, or_lmax, final_sweeps, ils_iters, skip_prob):
    if len(route) < 10:
        return route[:]

    deadline = time.monotonic() + max(0.03, budget_s)
    raw_perm = route_to_perm(route, nodes)
    perm = ensure_valid_perm(raw_perm, len(nodes), D)

    perm = two_opt_dlb(perm, D, cand, max_sweeps=initial_sweeps, skip_prob=skip_prob, deadline=deadline)
    if or_lmax > 0 and time.monotonic() < deadline:
        perm = or_opt_move(perm, D, cand, Lmax=or_lmax, deadline=deadline)
    if final_sweeps > 0 and time.monotonic() < deadline:
        perm = two_opt_dlb(perm, D, cand, max_sweeps=final_sweeps, skip_prob=max(0.0, skip_prob * 0.5), deadline=deadline)

    best = perm[:]
    best_len = tour_len_from_perm(best, D)
    for ils_iter in range(max(0, ils_iters)):
        if time.monotonic() > deadline:
            break
        record_event('ils_double_bridge')
        kick = double_bridge_on_perm(best)
        kick = ensure_valid_perm(kick, len(nodes), D)
        kick = two_opt_dlb(kick, D, cand, max_sweeps=2, skip_prob=max(0.0, skip_prob * 0.5), deadline=deadline)
        if time.monotonic() < deadline and or_lmax > 0:
            kick = or_opt_move(kick, D, cand, Lmax=min(2, or_lmax), deadline=deadline)
        if time.monotonic() < deadline:
            kick = two_opt_dlb(kick, D, cand, max_sweeps=1, skip_prob=0.0, deadline=deadline)

        L = tour_len_from_perm(kick, D)
        if L < best_len:
            best, best_len = kick[:], L
            record_event(f'ils_improvement_{ils_iter}')
    return perm_to_route(best, nodes)

def polish_light(nodes, route, D=None, cand=None):
    if len(route) < 10:
        return route[:]
    if D is None:
        D = build_distance_matrix(nodes)
    if cand is None:
        cand = build_candidate_sets(nodes, m=CAND_M, knn_k=CAND_KNN, use_delaunay=USE_DELAUNAY)
    return _polish_with_config(
        nodes,
        route,
        D,
        cand,
        budget_s=TIME_BUDGET_S * LIGHT_POLISH_FRACTION,
        initial_sweeps=LIGHT_MAX_SWEEPS_2OPT,
        or_lmax=LIGHT_OR_OPT_L,
        final_sweeps=1,
        ils_iters=0,
        skip_prob=SKIP_PROB_2OPT,
    )

def polish_fast(nodes, route, D=None, cand=None):
    """Pulido con presupuesto de tiempo: 2-opt(DLB) → Or-opt → 2-opt(DLB) → ILS."""
    record_event('polish_fast_entry')
    if len(route) < 10:
        record_event('polish_fast_short_exit')
        return route[:]
    if D is None:
        D = build_distance_matrix(nodes)
    if cand is None:
        cand = build_candidate_sets(nodes, m=CAND_M, knn_k=CAND_KNN, use_delaunay=USE_DELAUNAY)

    best_route = _polish_with_config(
        nodes,
        route,
        D,
        cand,
        budget_s=TIME_BUDGET_S,
        initial_sweeps=MAX_SWEEPS_2OPT,
        or_lmax=OR_OPT_MAX_L,
        final_sweeps=2,
        ils_iters=ILS_ITERS,
        skip_prob=max(0.0, SKIP_PROB_2OPT * 0.33),
    )
    record_event('polish_fast_exit')
    return best_route
def _concat_routes_q_order(routes):
    order = Q_ORDER
    seq = []
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

def _frontier_mid(direction, xmid, ymid):
    return xmid if direction == 'vertical' else ymid

def _enumerate_frontier_choice_maps(frontier_options, max_count=MAX_ASSEMBLY_PER_PARTITION):
    option_lists = []
    edge_keys = []
    for q1, q2, direction in ADJACENCY_CYCLE:
        edge_key = (q1, q2)
        edge_keys.append(edge_key)
        option_lists.append(frontier_options.get(edge_key) or [((None, None), float('inf'))])

    scored = []
    for combo in product(*option_lists):
        score = sum(item[1] for item in combo if np.isfinite(item[1]))
        choice_map = {edge_key: item for edge_key, item in zip(edge_keys, combo)}
        scored.append((score, choice_map))
    scored.sort(key=lambda item: item[0])
    return [choice_map for _, choice_map in scored[:max(1, max_count)]]

def _build_assembly_candidate(quads, frontier_choice_map, max_iter_local):
    routes = {}
    connections = []
    anchor_map = {
        'Q1': (('Q3', 'Q1'), ('Q1', 'Q2')),
        'Q2': (('Q1', 'Q2'), ('Q2', 'Q4')),
        'Q4': (('Q2', 'Q4'), ('Q4', 'Q3')),
        'Q3': (('Q4', 'Q3'), ('Q3', 'Q1')),
    }

    def _frontier_node(edge_key, idx):
        pair = frontier_choice_map.get(edge_key)
        if not pair:
            return None
        (u, v), _ = pair
        return u if idx == 0 else v

    for edge_key, pair in frontier_choice_map.items():
        (a, b), _ = pair
        if a is not None and b is not None:
            connections.append((a, b))

    for q in Q_ORDER:
        pts = quads[q]
        if len(pts) > 1:
            prev_key, next_key = anchor_map[q]
            start_anchor = _frontier_node(prev_key, 1)
            end_anchor = _frontier_node(next_key, 0)
            routes[q] = build_anchored_path(
                pts,
                start_anchor,
                end_anchor,
                max_iter=max(10, min(100, max_iter_local)),
            )
        else:
            routes[q] = pts[:]

    initial_tour = []
    for q in Q_ORDER:
        initial_tour.extend(routes.get(q, []))
    return routes, connections, initial_tour

def gsph_fc(nodes, MAX_ITER_LOCAL=800 ,EPS_FRONTIER=5):
    record_event('gsph_fc_entry')
    if not nodes:
        record_event('gsph_fc_exit')
        return {'QALL': []}, [], 0, 0.0, 0.0

    D = build_distance_matrix(nodes)
    cand = build_candidate_sets(nodes, m=CAND_M, knn_k=CAND_KNN, use_delaunay=USE_DELAUNAY)
    candidate_pool = []

    record_event('candidate_generation_start')
    for part_idx, (part_label, xmid, ymid) in enumerate(generate_partition_portfolio(nodes), start=1):
        quads, xmid, ymid = subdivide_quadrants(nodes, xmid=xmid, ymid=ymid)
        frontier_options = {}
        for q1, q2, direction in ADJACENCY_CYCLE:
            frontier_options[(q1, q2)] = best_frontier_pairs(
                quads[q1],
                quads[q2],
                direction,
                _frontier_mid(direction, xmid, ymid),
                EPS_FRONTIER,
                k=FRONTIER_TOPK,
            )

        for choice_map in _enumerate_frontier_choice_maps(frontier_options, max_count=MAX_ASSEMBLY_PER_PARTITION):
            routes, connections, initial_tour = _build_assembly_candidate(quads, choice_map, MAX_ITER_LOCAL)
            if not initial_tour:
                continue
            light_tour = polish_light(nodes, initial_tour, D=D, cand=cand)
            candidate_pool.append({
                'partition_label': part_label,
                'partition_idx': part_idx,
                'xmid': xmid,
                'ymid': ymid,
                'connections': connections,
                'seed_routes': routes,
                'tour': light_tour,
                'light_len': total_tour_length(light_tour),
                'raw_len': total_tour_length(initial_tour),
            })
    record_event('candidate_generation_end')

    if not candidate_pool:
        quads, xmid, ymid = subdivide_quadrants(nodes)
        routes = {q: quads[q][:] for q in Q_ORDER}
        initial_tour = _concat_routes_q_order(routes)
        candidate_pool.append({
            'partition_label': 'fallback',
            'partition_idx': 0,
            'xmid': xmid,
            'ymid': ymid,
            'connections': [],
            'seed_routes': routes,
            'tour': initial_tour,
            'light_len': total_tour_length(initial_tour),
            'raw_len': total_tour_length(initial_tour),
        })

    candidate_pool.sort(key=lambda item: (item['light_len'], item['raw_len']))
    finalists = candidate_pool[:max(1, min(ASSEMBLY_FINALISTS, len(candidate_pool)))]

    best = None
    record_event('polish_start')
    for finalist in finalists:
        improved_tour = polish_fast(nodes, finalist['tour'], D=D, cand=cand)
        total_len = total_tour_length(improved_tour)
        if best is None or total_len < best['total_len']:
            best = {
                'tour': improved_tour,
                'total_len': total_len,
                'connections': finalist['connections'],
                'xmid': finalist['xmid'],
                'ymid': finalist['ymid'],
            }
    record_event('polish_end')

    improved_tour = best['tour'] if best else nodes[:]
    total_len = best['total_len'] if best else total_tour_length(improved_tour)
    routes_out = {'QALL': improved_tour}
    record_event('gsph_fc_exit')
    return routes_out, (best['connections'] if best else []), total_len, (best['xmid'] if best else 0.0), (best['ymid'] if best else 0.0)

def plot_gsph_fc(routes, conns, xmid, ymid, save_path=None):
    if plt is None:
        return
    plt.figure(figsize=(10,10))
    
    all_points = []
    for r in routes.values():
        all_points.extend(r)
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # Add some padding
        padding = max((xmax-xmin), (ymax-ymin)) * 0.1
        xmin -= padding; xmax += padding
        ymin -= padding; ymax += padding
    else:
        xmin, xmax = xmid - 100, xmid + 100
        ymin, ymax = ymid - 100, ymid + 100
    

    alpha = 0.08 
    plt.fill_between([xmin, xmid], [ymid, ymid], [ymax, ymax], color='blue', alpha=alpha)
    plt.fill_between([xmid, xmax], [ymid, ymid], [ymax, ymax], color='green', alpha=alpha)
    plt.fill_between([xmin, xmid], [ymin, ymin], [ymid, ymid], color='red', alpha=alpha)
    plt.fill_between([xmid, xmax], [ymin, ymin], [ymid, ymid], color='orange', alpha=alpha)
    
    fontsize = 14
    plt.text(xmin + (xmid-xmin)/2, ymid + (ymax-ymid)/2, 'Q1 (NW)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmid + (xmax-xmid)/2, ymid + (ymax-ymid)/2, 'Q2 (NE)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmin + (xmid-xmin)/2, ymin + (ymid-ymin)/2, 'Q3 (SW)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmid + (xmax-xmid)/2, ymin + (ymid-ymin)/2, 'Q4 (SE)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    
    plt.axvline(xmid, linestyle='-', color='black', alpha=0.5, linewidth=1.5)
    plt.axhline(ymid, linestyle='-', color='black', alpha=0.5, linewidth=1.5)
    
    colors = ['blue', 'green', 'red', 'orange']
    for idx, (q, r) in enumerate(routes.items()):
        if len(r) > 1:
            x, y = zip(*r)
            plt.plot(x, y, marker='o', color=colors[idx % len(colors)], 
                     linewidth=2, markersize=4, label=q)
            if q == 'QALL':
                plt.plot([x[-1], x[0]], [y[-1], y[0]], 
                         color=colors[idx % len(colors)], linewidth=2)
        elif len(r) == 1:
            plt.scatter(*r[0], color='black', marker='x')
            
    if conns:
        for i, (a, b) in enumerate(conns):
            plt.plot([a[0], b[0]], [a[1], b[1]], 
                     linestyle='--', color='purple', linewidth=1.5, alpha=0.8)
            plt.scatter([a[0], b[0]], [a[1], b[1]], color='purple', s=40, 
                        marker='D', edgecolors='white', linewidth=1)
            
    plt.title("GSPH–FC (perfil rápido)", fontsize=16)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3); plt.axis('equal')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _plot_with_title(routes, conns, xmid, ymid, save_path, title_suffix=""):
    """Versión de plot_gsph_fc con título personalizado para múltiples runs."""
    if plt is None:
        return
    plt.figure(figsize=(10,10))
    
    all_points = []
    for r in routes.values():
        all_points.extend(r)
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        padding = max((xmax-xmin), (ymax-ymin)) * 0.1
        xmin -= padding; xmax += padding
        ymin -= padding; ymax += padding
    else:
        xmin, xmax = xmid - 100, xmid + 100
        ymin, ymax = ymid - 100, ymid + 100
    
    alpha = 0.08 
    plt.fill_between([xmin, xmid], [ymid, ymid], [ymax, ymax], color='blue', alpha=alpha)
    plt.fill_between([xmid, xmax], [ymid, ymid], [ymax, ymax], color='green', alpha=alpha)
    plt.fill_between([xmin, xmid], [ymin, ymin], [ymid, ymid], color='red', alpha=alpha)
    plt.fill_between([xmid, xmax], [ymin, ymin], [ymid, ymid], color='orange', alpha=alpha)
    
    fontsize = 14
    plt.text(xmin + (xmid-xmin)/2, ymid + (ymax-ymid)/2, 'Q1 (NW)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmid + (xmax-xmid)/2, ymid + (ymax-ymid)/2, 'Q2 (NE)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmin + (xmid-xmin)/2, ymin + (ymid-ymin)/2, 'Q3 (SW)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    plt.text(xmid + (xmax-xmid)/2, ymin + (ymid-ymin)/2, 'Q4 (SE)', 
             ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.7)
    
    plt.axvline(xmid, linestyle='-', color='black', alpha=0.5, linewidth=1.5)
    plt.axhline(ymid, linestyle='-', color='black', alpha=0.5, linewidth=1.5)
    
    colors = ['blue', 'green', 'red', 'orange']
    for idx, (q, r) in enumerate(routes.items()):
        if len(r) > 1:
            x, y = zip(*r)
            plt.plot(x, y, marker='o', color=colors[idx % len(colors)], 
                     linewidth=2, markersize=4, label=q)
            if q == 'QALL':
                plt.plot([x[-1], x[0]], [y[-1], y[0]], 
                         color=colors[idx % len(colors)], linewidth=2)
        elif len(r) == 1:
            plt.scatter(*r[0], color='black', marker='x')
            
    if conns:
        for i, (a, b) in enumerate(conns):
            plt.plot([a[0], b[0]], [a[1], b[1]], 
                     linestyle='--', color='purple', linewidth=1.5, alpha=0.8)
            plt.scatter([a[0], b[0]], [a[1], b[1]], color='purple', s=40, 
                        marker='D', edgecolors='white', linewidth=1)
    
    title = f"GSPH–FC{title_suffix}"
    plt.title(title, fontsize=14)
    plt.xlabel("Coordenada X", fontsize=12)
    plt.ylabel("Coordenada Y", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3); plt.axis('equal')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _create_comparison_plot(all_runs_data, inst_name, save_path, best_run_idx):
    """Genera un grid comparativo de todos los runs."""
    if plt is None:
        return
    import matplotlib.gridspec as gridspec
    
    n_runs = len(all_runs_data)
    if n_runs == 0:
        return
    
    ncols = min(3, n_runs)
    nrows = (n_runs + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(6*ncols, 5*nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, run_data in enumerate(all_runs_data):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        routes = run_data['routes']
        xmid = run_data['xm']
        ymid = run_data['ym']
        length = run_data['total_length']
        run_idx = run_data['run_idx']
        is_best = (run_idx == best_run_idx)
        
        for r in routes.values():
            if len(r) > 1:
                x, y = zip(*r)
                color = 'red' if is_best else 'blue'
                linewidth = 2.5 if is_best else 1.5
                ax.plot(x, y, marker='o', color=color, linewidth=linewidth, 
                       markersize=3, alpha=0.8)
                ax.plot([x[-1], x[0]], [y[-1], y[0]], color=color, 
                       linewidth=linewidth, alpha=0.8)
        
        title = f"Run {run_idx+1}\nLength: {length:.1f}"
        if is_best:
            title += " ⭐"
            ax.set_facecolor('#ffffcc')
        
        ax.set_title(title, fontsize=10, fontweight='bold' if is_best else 'normal')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    fig.suptitle(f'Comparación de {n_runs} ejecuciones - {inst_name}', 
                 fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == "__main__":
    import csv
    import threading
    import math
    try:
        import matplotlib.pyplot as _plt
    except Exception:
        _plt = None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances_dir = os.path.abspath(BASE_DIR)
    if not os.path.isdir(instances_dir):
        print(f"Instances directory not found: {instances_dir}")
        raise SystemExit(1)

    files = [f for f in os.listdir(instances_dir) if f.lower().endswith('.tsp')]
    files.sort()

    target_arg = None
    profile_flag = False
    if len(sys.argv) > 1:
        args = [a for a in sys.argv[1:]]
        if '--profile' in args:
            profile_flag = True
            args.remove('--profile')
        if args:
            target_arg = args[0]
        if target_arg in ('-h', '--help'):
            print('Usage: python GSPH_BASE.py [instance_name|file.tsp]')
            print('If no argument is given, all instances under', instances_dir, 'are processed.')
            raise SystemExit(0)

    if target_arg:
        if target_arg.lower().endswith('.tsp'):
            candidate = target_arg if os.path.isabs(target_arg) else os.path.join(instances_dir, target_arg)
            if os.path.isfile(candidate):
                files = [os.path.basename(candidate)]
            else:
                print('Instance file not found:', candidate)
                raise SystemExit(1)
        else:
            candidate = target_arg + '.tsp'
            if candidate in files:
                files = [candidate]
            else:
                matches = [f for f in files if target_arg.lower() in f.lower()]
                if len(matches) == 1:
                    files = matches
                elif len(matches) > 1:
                    print('Multiple matches for', target_arg, ':', matches)
                    raise SystemExit(1)
                else:
                    print('No instance found matching', target_arg)
                    raise SystemExit(1)

    limit = os.getenv('GSPH_LIMIT')
    try:
        limit = int(limit) if limit else None
    except Exception:
        limit = None
    if limit:
        files = files[:limit]

    summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance', 'n_nodes', 'total_length', 'time_seconds', 'route_png', 'notes'])

    resources_summary = []

    def plot_resource_times(times, rss, cpu, outpath):
        if not times: return
        if _plt is None:
            return
        fig, ax1 = _plt.subplots(figsize=(8,4))
        ax1.plot(times, [r/1024/1024 for r in rss], 'r-', label='RSS (MB)')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('RSS (MB)', color='r')
        ax2 = ax1.twinx()
        ax2.plot(times, cpu, 'b--', label='CPU%')
        ax2.set_ylabel('CPU %', color='b')
        fig.tight_layout()
        _plt.savefig(outpath, bbox_inches='tight')
        _plt.close()

    for fname in files:
        inst_name = os.path.splitext(fname)[0]
        outdir = os.path.join(RESULTS_DIR, inst_name)
        os.makedirs(outdir, exist_ok=True)
        tsp_path = os.path.join(instances_dir, fname)
        try:
            nodes = read_tsplib(tsp_path)
        except Exception as e:
            with open(os.path.join(outdir, 'error.txt'), 'w', encoding='utf-8') as ef:
                ef.write(str(e))
            with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([inst_name, 0, '', '', os.path.join(outdir, 'error.txt'), 'read_error'])
            print(f"Skipped {fname}: read error")
            continue

        num_runs = PROFILE_RUNS if profile_flag else 1
        all_results = []
        all_times = []
        all_runs_data = []  # Guardar datos de TODAS las ejecuciones
        best_result = None
        best_length = float('inf')
        
        print(f"Processing {inst_name} ({num_runs} run{'s' if num_runs > 1 else ''})...")
        
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"  Run {run_idx + 1}/{num_runs}...", end="")
            
            t0 = time.time()
            if profile_flag and run_idx == 0:  # Solo para la primera ejecución
                event_enable(t0)
            
            sample_times = []
            sample_rss = []
            sample_cpu = []
            stop_event = threading.Event()
            proc = psutil.Process() if psutil else None

            def monitor(interval=0.05):
                while not stop_event.is_set():
                    try:
                        now = time.time() - t0
                        rss = proc.memory_info().rss if proc else 0
                        cpu = proc.cpu_percent(interval=None) if proc else 0
                        sample_times.append(now)
                        sample_rss.append(rss)
                        sample_cpu.append(cpu)
                    except Exception:
                        pass
                    time.sleep(interval)

            mon_thread = None
            if profile_flag and run_idx == 0 and psutil:  # Solo primera ejecución
                mon_thread = threading.Thread(target=monitor, daemon=True)
                mon_thread.start()

            try:
                if profile_flag and run_idx == 0 and tracemalloc:  # Solo primera ejecución
                    tracemalloc.start()
                if profile_flag and run_idx == 0 and tracker:  # Solo primera ejecución
                    mem_tracker = tracker.SummaryTracker()
                
                routes, conns, total_length, xm, ym = gsph_fc(nodes, MAX_ITER_LOCAL, EPS_FRONTIER)
                
                if profile_flag and run_idx == 0 and tracemalloc:
                    snap = tracemalloc.take_snapshot()
                    top = snap.statistics('lineno')[:10]
                    with open(os.path.join(outdir, 'tracemalloc_top.txt'), 'w', encoding='utf-8') as tf:
                        for stat in top:
                            tf.write(str(stat) + '\n')
                if profile_flag and run_idx == 0 and tracker:
                    # record a pympler summary (approx)
                    with open(os.path.join(outdir, 'pympler_summary.txt'), 'w', encoding='utf-8') as pf:
                        for t in mem_tracker.format_diff():
                            pf.write(str(t) + '\n')
            except Exception as e:
                with open(os.path.join(outdir, 'error.txt'), 'w', encoding='utf-8') as ef:
                    ef.write(str(e))
                with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([inst_name, len(nodes), '', '', os.path.join(outdir, 'error.txt'), 'run_error'])
                print(f"Failed {fname}: runtime error")
                break
            
            elapsed = time.time() - t0
            
            all_results.append(total_length)
            all_times.append(elapsed)
            run_data = {
                'run_idx': run_idx,
                'routes': routes,
                'conns': conns,
                'total_length': total_length,
                'xm': xm,
                'ym': ym,
                'elapsed': elapsed
            }
            all_runs_data.append(run_data)
            
            if total_length < best_length:
                best_length = total_length
                best_result = {
                    'routes': routes,
                    'conns': conns,
                    'total_length': total_length,
                    'xm': xm,
                    'ym': ym,
                    'elapsed': elapsed,
                    'sample_times': sample_times if run_idx == 0 else [],
                    'sample_rss': sample_rss if run_idx == 0 else [],
                    'sample_cpu': sample_cpu if run_idx == 0 else [],
                    'best_run_idx': run_idx,
                    'sample_cpu': sample_cpu if run_idx == 0 else []
                }
            
            if mon_thread:
                stop_event.set()
                mon_thread.join(timeout=1.0)
                
            if num_runs > 1:
                print(f" length={total_length:.2f}, time={elapsed:.3f}s")
        
        if best_result is None:
            continue
            
        routes = best_result['routes']
        conns = best_result['conns'] 
        totcompal_length = best_result['total_length']
        xm = best_result['xm']
        ym = best_result['ym']
        elapsed = best_result['elapsed']
        sample_times = best_result['sample_times']
        sample_rss = best_result['sample_rss'] 
        sample_cpu = best_result['sample_cpu']
        
        if num_runs > 1:
            best_run_idx = best_result.get('best_run_idx', 0)
            print(f"  Generando {len(all_runs_data)} gráficos...")
            for run_data in all_runs_data:
                idx = run_data['run_idx']
                is_best = (idx == best_run_idx)
                suffix = "_MEJOR" if is_best else ""
                png_name = f"grafico_{inst_name}_run_{idx+1:02d}{suffix}.png"
                png_path_run = os.path.join(outdir, png_name)
                
                title_suffix = f" | Run {idx+1}/{num_runs} - Length: {run_data['total_length']:.2f}"
                if is_best:
                    title_suffix += " MEJOR"
                
                _plot_with_title(run_data['routes'], run_data['conns'], 
                                run_data['xm'], run_data['ym'], 
                                png_path_run, title_suffix)
            
            png_comparison = os.path.join(outdir, f"grafico_{inst_name}_comparison.png")
            _create_comparison_plot(all_runs_data, inst_name, png_comparison, best_run_idx)
        
        png_path = os.path.join(outdir, f"grafico_{inst_name}.png")
        
        if profile_flag and sample_times:
            res_csv = os.path.join(outdir, 'resources.csv')
            with open(res_csv, 'w', newline='', encoding='utf-8') as rc:
                rwriter = csv.writer(rc)
                rwriter.writerow(['time_s', 'rss_bytes', 'cpu_percent'])
                for a,b,c in zip(sample_times, sample_rss, sample_cpu):
                    rwriter.writerow([f"{a:.4f}", int(b), f"{c:.2f}"])
            plot_resource_times(sample_times, sample_rss, sample_cpu, os.path.join(outdir, 'resources.png'))
        peak_rss = max(sample_rss) if sample_rss else (proc.memory_info().rss if proc else 0)

        res_txt = os.path.join(outdir, 'resultados.txt')
        with open(res_txt, 'w', encoding='utf-8') as f:
            f.write(f"Instancia: {fname}\n")
            f.write(f"N nodes: {len(nodes)}\n")
            if num_runs > 1:
                f.write(f"Número de ejecuciones: {num_runs}\n")
                f.write(f"MEJOR longitud de ruta: {total_length:.2f}\n")
                f.write(f"Longitud promedio     : {sum(all_results)/len(all_results):.2f}\n")
                f.write(f"Rango (min-max)       : {min(all_results):.2f} - {max(all_results):.2f}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"GRÁFICOS GENERADOS:\n")
                f.write(f"{'='*50}\n")
                best_run_idx = best_result.get('best_run_idx', 0)
                for run_data in all_runs_data:
                    idx = run_data['run_idx']
                    is_best = (idx == best_run_idx)
                    marker = "⭐ MEJOR" if is_best else ""
                    f.write(f"  • grafico_{inst_name}_run_{idx+1:02d}{'_MEJOR' if is_best else ''}.png - Length: {run_data['total_length']:.2f} {marker}\n")
                f.write(f"  • grafico_{inst_name}_MEJOR.png - Mejor resultado consolidado\n")
            else:
                f.write(f"Longitud total de la ruta: {total_length:.2f}\n")
            f.write(f"Tiempo de ejecución      : {elapsed:.3f} segundos\n")
            if profile_flag:
                events_file = os.path.join(outdir, 'events.csv')
                try:
                    with open(events_file, 'w', newline='', encoding='utf-8') as ef:
                        ew = csv.writer(ef)
                        ew.writerow(['time_s','event'])
                        for t,e in _events:
                            ew.writerow([f"{t:.6f}", e])
                    try:
                        if _plt is None:
                            raise RuntimeError("matplotlib unavailable")
                        times = [t for t,_ in _events]
                        fig, ax = _plt.subplots(figsize=(10, max(2, len(times)*0.15)))
                        ax.hlines(1, 0, max(times) if times else 1, color='white')
                        for i, (tt, ev) in enumerate(_events):
                            ax.plot(tt, 1, marker='o')
                            ax.text(tt, 1.02, ev, rotation=45, ha='left', va='bottom', fontsize=8)
                        ax.set_yticks([])
                        ax.set_xlabel('time (s)')
                        _plt.tight_layout()
                        _plt.savefig(os.path.join(outdir, 'events_timeline.png'), bbox_inches='tight')
                        _plt.close()
                    except Exception:
                        pass
                finally:
                    event_disable()
            f.write(f"Peak RSS (bytes)         : {int(peak_rss)}\n")

        notes = 'ok'
        if num_runs > 1:
            stats_file = os.path.join(outdir, 'multiple_runs_stats.txt')
            with open(stats_file, 'w', encoding='utf-8') as sf:
                sf.write(f"Estadísticas de {num_runs} ejecuciones:\n")
                sf.write(f"Mejor longitud:     {min(all_results):.2f}\n")
                sf.write(f"Peor longitud:      {max(all_results):.2f}\n")
                sf.write(f"Longitud promedio:  {sum(all_results)/len(all_results):.2f}\n")
                if len(all_results) > 1:
                    variance = sum((x - sum(all_results)/len(all_results))**2 for x in all_results) / (len(all_results) - 1)
                    std_dev = math.sqrt(variance)
                    sf.write(f"Desviación estándar: {std_dev:.2f}\n")
                sf.write(f"Mejor tiempo:       {min(all_times):.3f}s\n")
                sf.write(f"Peor tiempo:        {max(all_times):.3f}s\n")
                sf.write(f"Tiempo promedio:    {sum(all_times)/len(all_times):.3f}s\n")
                sf.write(f"\nDetalle de todas las ejecuciones:\n")
                for i, (length, time_taken) in enumerate(zip(all_results, all_times)):
                    marker = " (MEJOR)" if length == min(all_results) else ""
                    sf.write(f"Run {i+1}: length={length:.2f}, time={time_taken:.3f}s{marker}\n")
            notes = f'best_of_{num_runs}_runs'

        with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if num_runs > 1:
                avg_length = sum(all_results)/len(all_results)
                std_dev = math.sqrt(sum((x - avg_length)**2 for x in all_results) / (len(all_results) - 1)) if len(all_results) > 1 else 0
                writer.writerow([inst_name, len(nodes), f"{total_length:.2f}", f"{elapsed:.3f}", png_path, f'{notes}|avg={avg_length:.2f}|std={std_dev:.2f}'])
            else:
                writer.writerow([inst_name, len(nodes), f"{total_length:.2f}", f"{elapsed:.3f}", png_path, notes])

        if num_runs > 1:
            print(f"Processed {inst_name}: nodes={len(nodes)}, BEST={total_length:.2f} (avg={sum(all_results)/len(all_results):.2f}), best_time={elapsed:.3f}s")
        else:
            print(f"Processed {inst_name}: nodes={len(nodes)}, length={total_length:.2f}, time={elapsed:.2f}s")

    print('\nAll done. Summary written to', summary_path)
