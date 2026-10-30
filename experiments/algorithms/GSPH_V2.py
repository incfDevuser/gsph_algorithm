"""GSPH-FC V2: misma idea geométrica que GSPH_BASE (portafolio de particiones,
fronteras, caminos anclados, embudo y pulido), con el motor reescrito:

- Todo el pipeline trabaja con índices de nodo (no con tuplas de coordenadas).
- Búsqueda local (2-opt en ambas direcciones + Or-opt) e ILS compilados con
  numba sobre arreglos. Sin numba, el mismo código corre en Python puro.
- El ILS para por estancamiento (kicks sin mejora), no por tiempo fijo.
"""
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
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        # Respaldo sin numba: el decorador no hace nada y los kernels corren en Python.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco

EPS_FRONTIER      = 5
MAX_ITER_LOCAL    = 800
RESULTS_DIR       = "../results/v2"
BASE_DIR          = "../instances/standard"

CAND_M            = 20
CAND_KNN          = 25
USE_DELAUNAY      = True
OR_OPT_MAX_L      = 3     # Largo máximo de segmento en Or-opt

FRONTIER_TOPK               = 3
MAX_ASSEMBLY_PER_PARTITION  = 8
ASSEMBLY_FINALISTS          = 3
PARTITION_PORTFOLIO_LIMIT   = 6
PARTITION_SHIFT_FRAC        = 0.12
LIGHT_POLISH_FRACTION       = 0.20
RAW_PREFILTER_PER_PARTITION = 2
LIGHT_POOL_CAP              = 8
FINALIST_MARGIN_RATIO       = 1.01

# ILS: double-bridge local y parada por estancamiento.
DB_WINDOW_MIN     = 50     # Largo máximo de cada segmento del double-bridge: max(MIN, n / DIV)
DB_WINDOW_DIV     = 8
ILS_BLOCK         = 1000   # Kicks por llamada al kernel (entre chequeos de reloj)
JOURNAL_SIZE      = 4096   # Inversiones anotadas por kick para poder deshacerlo
ACCEPT_TEMP       = float(os.getenv('GSPH_TEMP', '0.2'))  # Temperatura de aceptación, en largos promedio de arista
STALL_MIN         = 5000   # Mínimo de kicks sin mejora antes de parar
STALL_FACTOR      = 300    # Kicks sin mejora = max(STALL_MIN, STALL_FACTOR * n)

def _env_int(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except Exception:
        return default

PROFILE_RUNS = _env_int('GSPH_PROFILE_RUNS', 10)

# Tope de tiempo de seguridad por run (segundos); la parada normal es por estancamiento.
TIME_CAP_S = float(os.getenv('GSPH_TIME_CAP', '60'))

def stall_limit(n_nodes):
    """Kicks consecutivos sin mejora tras los cuales el ILS se detiene."""
    override = os.getenv('GSPH_STALL')
    if override:
        try:
            return max(1, int(override))
        except Exception:
            pass
    return max(STALL_MIN, STALL_FACTOR * n_nodes)

def polish_cand_knn(n_nodes):
    """kNN de candidatos según tamaño."""
    if n_nodes > 250:
        return 40
    return CAND_KNN

BIG_COST = 1 << 62

Q_ORDER = ['Q1', 'Q2', 'Q4', 'Q3']
ADJACENCY_CYCLE = [
    ('Q1', 'Q2', 'vertical'),
    ('Q2', 'Q4', 'horizontal'),
    ('Q4', 'Q3', 'vertical'),
    ('Q3', 'Q1', 'horizontal'),
]


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

def get_search_profile(n_nodes):
    if n_nodes <= 150:
        return {
            'partition_limit': min(PARTITION_PORTFOLIO_LIMIT, 5),
            'assembly_limit': min(MAX_ASSEMBLY_PER_PARTITION, 5),
            'per_partition_prefilter': max(1, RAW_PREFILTER_PER_PARTITION),
            'light_pool_cap': min(LIGHT_POOL_CAP, 8),
            'finalists': min(2, ASSEMBLY_FINALISTS),
            'light_fraction': min(LIGHT_POLISH_FRACTION, 0.12),
        }
    if n_nodes <= 300:
        return {
            'partition_limit': min(PARTITION_PORTFOLIO_LIMIT, 4),
            'assembly_limit': min(MAX_ASSEMBLY_PER_PARTITION, 4),
            'per_partition_prefilter': 2,
            'light_pool_cap': min(LIGHT_POOL_CAP, 6),
            'finalists': min(2, ASSEMBLY_FINALISTS),
            'light_fraction': min(LIGHT_POLISH_FRACTION, 0.08),
        }
    return {
        'partition_limit': min(PARTITION_PORTFOLIO_LIMIT, 3),
        'assembly_limit': min(MAX_ASSEMBLY_PER_PARTITION, 3),
        'per_partition_prefilter': 1,
        'light_pool_cap': min(LIGHT_POOL_CAP, 4),
        'finalists': 1,
        'light_fraction': min(LIGHT_POLISH_FRACTION, 0.05),
    }


def build_distance_matrix(points, block=1024):
    """Matriz EUC_2D entera, calculada por bloques de filas para no crear temporales n x n x 2."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    D = np.empty((n, n), dtype=np.int32)
    for s in range(0, n, block):
        diff = P[s:s + block, None, :] - P[None, :, :]
        D[s:s + block] = np.asarray(np.sqrt((diff**2).sum(-1)) + 0.5, dtype=np.int32)
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


# ---------------------------------------------------------------------------
# Kernels compilados (numba). Tour = arreglo de nodos, pos[nodo] = posición.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _seed_kernels(seed):
    np.random.seed(seed)

@njit(cache=True)
def _tour_len(tour, D):
    n = tour.shape[0]
    s = 0
    for i in range(n - 1):
        s += D[tour[i], tour[i + 1]]
    s += D[tour[n - 1], tour[0]]
    return s

@njit(cache=True)
def _reverse(tour, pos, i, j):
    """Invierte el tramo circular de posiciones i..j; si es más corto, invierte el complemento."""
    n = tour.shape[0]
    L = (j - i) % n + 1
    if 2 * L > n:
        i, j = (j + 1) % n, (i - 1) % n
        L = n - L
    for k in range(L // 2):
        a = (i + k) % n
        b = (j - k) % n
        va = tour[a]
        vb = tour[b]
        tour[a] = vb
        pos[vb] = a
        tour[b] = va
        pos[va] = b

@njit(cache=True)
def _rev_logged(tour, pos, i, j, journal, jcount):
    """_reverse que además anota (i, j) para poder deshacerlo (una inversión es su propia inversa)."""
    _reverse(tour, pos, i, j)
    k = jcount[0]
    if k < journal.shape[0]:
        journal[k, 0] = i
        journal[k, 1] = j
    jcount[0] = k + 1

@njit(cache=True)
def _two_opt_move(tour, pos, a, b, c, d, journal, jcount):
    """Quita (a,b),(c,d) y agrega (a,c),(b,d); b sigue a 'a' y d sigue a c en el mismo sentido."""
    n = tour.shape[0]
    if tour[(pos[a] + 1) % n] == b:
        _rev_logged(tour, pos, pos[b], pos[c], journal, jcount)
    else:
        _rev_logged(tour, pos, pos[a], pos[d], journal, jcount)

@njit(cache=True)
def _or_move(tour, pos, p, s1, s2, nx, u, v, rev, journal, jcount):
    """Mueve el segmento s1..s2 (entre p y nx) a la arista (u,v) con 2 o 3 inversiones.

    rev=False deja (u, s1..s2, v); rev=True deja (u, s2..s1, v).
    """
    if u == nx:
        _two_opt_move(tour, pos, p, s1, nx, v, journal, jcount)
    elif v == p:
        _two_opt_move(tour, pos, u, p, s2, nx, journal, jcount)
    else:
        _two_opt_move(tour, pos, p, s1, u, v, journal, jcount)
        _two_opt_move(tour, pos, p, u, nx, s2, journal, jcount)
    # En este punto queda (u, s2..s1, v).
    if not rev and s1 != s2:
        _two_opt_move(tour, pos, u, s2, s1, v, journal, jcount)

@njit(cache=True)
def _push(v, qbuf, inq, head, cnt):
    if inq[v] == 0:
        inq[v] = 1
        qbuf[(head + cnt) % qbuf.shape[0]] = v
        cnt += 1
    return cnt

@njit(cache=True)
def _local_search(tour, pos, D, cand, ncand, init, ninit, qbuf, inq, journal, jcount, or_lmax):
    """2-opt (sucesor y predecesor) + Or-opt con listas de vecinos y cola de nodos activos.

    Solo procesa los nodos de 'init' y los que reactivan los movimientos (equivale
    a don't-look bits). Devuelve la ganancia total, para llevar la longitud sin
    recalcularla.
    """
    n = tour.shape[0]
    head = 0
    cnt = 0
    for t in range(ninit):
        cnt = _push(init[t], qbuf, inq, head, cnt)
    total = 0
    while cnt > 0:
        a = qbuf[head]
        head = (head + 1) % n
        cnt -= 1
        inq[a] = 0

        pa = pos[a]
        succ_a = tour[(pa + 1) % n]
        pred_a = tour[(pa - 1) % n]
        d_as = D[a, succ_a]
        d_ap = D[a, pred_a]
        best_gain = 0
        mtype = 0
        m1 = -1
        m2 = -1
        mL = 0
        mrev = False

        # 2-opt: la arista nueva (a,c) debe ser más corta que la que sale de a.
        for k in range(ncand[a]):
            c = cand[a, k]
            dac = D[a, c]
            if dac >= d_as and dac >= d_ap:
                break
            pc = pos[c]
            if dac < d_as:
                sc = tour[(pc + 1) % n]
                if c != succ_a and sc != a:
                    g = d_as + D[c, sc] - dac - D[succ_a, sc]
                    if g > best_gain:
                        best_gain = g
                        mtype = 1
                        m1 = succ_a
                        m2 = c
            if dac < d_ap:
                pcn = tour[(pc - 1) % n]
                if c != pred_a and pcn != a:
                    g = d_ap + D[pcn, c] - dac - D[pred_a, pcn]
                    if g > best_gain:
                        best_gain = g
                        mtype = 2
                        m1 = a
                        m2 = pcn

        # Or-opt: segmentos de largo 1..or_lmax con a en un extremo.
        for L in range(1, or_lmax + 1):
            if L + 3 > n:
                break
            for side in range(2):
                if side == 1 and L == 1:
                    break
                if side == 0:
                    i = pa
                else:
                    i = (pa - L + 1) % n
                s1 = tour[i]
                s2 = tour[(i + L - 1) % n]
                p = tour[(i - 1) % n]
                nx = tour[(i + L) % n]
                g_rem = D[p, s1] + D[s2, nx] - D[p, nx]
                if g_rem <= best_gain:
                    continue
                for k in range(ncand[a]):
                    c = cand[a, k]
                    if D[a, c] >= g_rem - best_gain:
                        break
                    if (pos[c] - i) % n < L:
                        continue
                    # Inserción en (c, succ c) con a junto a c.
                    v = tour[(pos[c] + 1) % n]
                    if (pos[v] - i) % n >= L:
                        if a == s1:
                            g = g_rem - (D[c, s1] + D[s2, v] - D[c, v])
                            rv = False
                        else:
                            g = g_rem - (D[c, s2] + D[s1, v] - D[c, v])
                            rv = True
                        if g > best_gain:
                            best_gain = g
                            mtype = 3
                            m1 = s1
                            m2 = c
                            mL = L
                            mrev = rv
                    # Inserción en (pred c, c) con a junto a c.
                    u = tour[(pos[c] - 1) % n]
                    if (pos[u] - i) % n >= L:
                        if a == s2:
                            g = g_rem - (D[u, s1] + D[s2, c] - D[u, c])
                            rv = False
                        else:
                            g = g_rem - (D[u, s2] + D[s1, c] - D[u, c])
                            rv = True
                        if g > best_gain:
                            best_gain = g
                            mtype = 3
                            m1 = s1
                            m2 = u
                            mL = L
                            mrev = rv

        if mtype == 0:
            continue
        total += best_gain
        if mtype == 1 or mtype == 2:
            # Invierte m1..m2: (x, m1 .. m2, y) -> (x, m2 .. m1, y).
            x = tour[(pos[m1] - 1) % n]
            y = tour[(pos[m2] + 1) % n]
            _rev_logged(tour, pos, pos[m1], pos[m2], journal, jcount)
            cnt = _push(x, qbuf, inq, head, cnt)
            cnt = _push(m1, qbuf, inq, head, cnt)
            cnt = _push(m2, qbuf, inq, head, cnt)
            cnt = _push(y, qbuf, inq, head, cnt)
        else:
            i = pos[m1]
            s2 = tour[(i + mL - 1) % n]
            p = tour[(i - 1) % n]
            nx = tour[(i + mL) % n]
            v = tour[(pos[m2] + 1) % n]
            _or_move(tour, pos, p, m1, s2, nx, m2, v, mrev, journal, jcount)
            cnt = _push(p, qbuf, inq, head, cnt)
            cnt = _push(nx, qbuf, inq, head, cnt)
            cnt = _push(m1, qbuf, inq, head, cnt)
            cnt = _push(s2, qbuf, inq, head, cnt)
            cnt = _push(m2, qbuf, inq, head, cnt)
            cnt = _push(v, qbuf, inq, head, cnt)
        cnt = _push(a, qbuf, inq, head, cnt)
    return total

@njit(cache=True)
def _ils_block(tour, pos, D, cand, ncand, cur_len, best_len, kicks, window, stall, temp,
               qbuf, inq, journal, jcount, or_lmax, best, seg, ends):
    """Corre 'kicks' iteraciones de ILS: double-bridge local + búsqueda local acotada.

    Acepta si la longitud queda <= actual (mesetas incluidas) y, si empeora, con
    probabilidad exp(-delta/temp) (temp=0 => nunca). Al rechazar, deshace las
    inversiones anotadas en orden inverso y restaura la ventana del kick, sin
    copiar el tour completo. 'best' guarda el mejor tour visto (best_len).
    'stall' cuenta kicks seguidos sin mejorar best_len.
    """
    n = tour.shape[0]
    for _ in range(kicks):
        L1 = np.random.randint(1, window + 1)
        L2 = np.random.randint(1, window + 1)
        L3 = np.random.randint(1, window + 1)
        e0 = np.random.randint(0, n)
        e1 = e0 + L1
        e2 = e1 + L2
        e3 = e2 + L3
        a0 = tour[e0 % n]
        b0 = tour[(e0 + 1) % n]
        a1 = tour[e1 % n]
        b1 = tour[(e1 + 1) % n]
        a2 = tour[e2 % n]
        b2 = tour[(e2 + 1) % n]
        a3 = tour[e3 % n]
        b3 = tour[(e3 + 1) % n]
        delta = (D[a0, b2] + D[a3, b1] + D[a2, b0] + D[a1, b3]) - \
                (D[a0, b0] + D[a1, b1] + D[a2, b2] + D[a3, b3])
        # A B C Dd E -> A Dd C B E, reescribiendo solo la ventana e0+1..e3.
        W = L1 + L2 + L3
        for t in range(W):
            seg[t] = tour[(e0 + 1 + t) % n]
        q = e0 + 1
        for t in range(L1 + L2, W):
            v = seg[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        for t in range(L1, L1 + L2):
            v = seg[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        for t in range(L1):
            v = seg[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        ends[0] = a0
        ends[1] = b0
        ends[2] = a1
        ends[3] = b1
        ends[4] = a2
        ends[5] = b2
        ends[6] = a3
        ends[7] = b3
        jcount[0] = 0
        gain = _local_search(tour, pos, D, cand, ncand, ends, 8, qbuf, inq, journal, jcount, or_lmax)
        new_len = cur_len + delta - gain
        accept = new_len <= cur_len
        if not accept and temp > 0.0:
            accept = np.random.random() < np.exp(-(new_len - cur_len) / temp)
        if accept:
            cur_len = new_len
            if new_len < best_len:
                best_len = new_len
                stall = 0
                for t in range(n):
                    best[t] = tour[t]
            else:
                stall += 1
        else:
            k = jcount[0]
            if k <= journal.shape[0]:
                for t in range(k - 1, -1, -1):
                    _reverse(tour, pos, journal[t, 0], journal[t, 1])
                q = e0 + 1
                for t in range(W):
                    v = seg[t]
                    tour[q % n] = v
                    pos[v] = q % n
                    q += 1
            else:
                for t in range(n):
                    v = best[t]
                    tour[t] = v
                    pos[v] = t
                cur_len = best_len
            stall += 1
    return cur_len, best_len, stall

@njit(cache=True)
def _open_2opt(path, D, max_iter):
    """2-opt de camino abierto con extremos fijos (primera mejora), como tsp_2opt de BASE."""
    m = path.shape[0]
    for _ in range(max_iter):
        improved = False
        for i in range(1, m - 2):
            for k in range(i + 1, m - 1):
                old = D[path[i - 1], path[i]] + D[path[k], path[k + 1]]
                new = D[path[i - 1], path[k]] + D[path[i], path[k + 1]]
                if new < old:
                    lo = i
                    hi = k
                    while lo < hi:
                        tmp = path[lo]
                        path[lo] = path[hi]
                        path[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
        if not improved:
            break

@njit(cache=True)
def _anchored_path(nodes, s_loc, e_loc, D, max_iter):
    """Camino abierto de s a e: inserción de costo mínimo incremental + 2-opt.

    Cuando desaparece la mejor arista de un nodo pendiente, su costo viejo queda
    como cota inferior ('stale'): los costos de inserción solo pueden subir, así
    que basta recalcularlo cuando ese nodo llega a ser el mínimo (evaluación perezosa).
    """
    m = nodes.shape[0]
    nxt = np.full(m, -1, np.int64)
    nxt[s_loc] = e_loc
    in_path = np.zeros(m, np.bool_)
    in_path[s_loc] = True
    in_path[e_loc] = True
    best_u = np.full(m, s_loc, np.int64)
    best_c = np.zeros(m, np.int64)
    stale = np.zeros(m, np.bool_)
    gs = nodes[s_loc]
    ge = nodes[e_loc]
    for x in range(m):
        if not in_path[x]:
            gx = nodes[x]
            best_c[x] = D[gs, gx] + D[gx, ge] - D[gs, ge]
    big = BIG_COST
    for _ in range(m - 2):
        while True:
            r = -1
            bc = big
            for x in range(m):
                if not in_path[x] and best_c[x] < bc:
                    bc = best_c[x]
                    r = x
            if not stale[r]:
                break
            # La cota no es exacta: se recalcula sobre todo el camino y se vuelve a elegir.
            gx = nodes[r]
            bu = s_loc
            bcost = big
            w = s_loc
            while w != e_loc:
                wn = nxt[w]
                cst = D[nodes[w], gx] + D[gx, nodes[wn]] - D[nodes[w], nodes[wn]]
                if cst < bcost:
                    bcost = cst
                    bu = w
                w = wn
            best_u[r] = bu
            best_c[r] = bcost
            stale[r] = False
        u = best_u[r]
        v = nxt[u]
        nxt[u] = r
        nxt[r] = v
        in_path[r] = True
        gu = nodes[u]
        gv = nodes[v]
        gr = nodes[r]
        for x in range(m):
            if in_path[x]:
                continue
            gx = nodes[x]
            if not stale[x] and best_u[x] == u:
                stale[x] = True     # su mejor arista (u,v) desapareció
            c1 = D[gu, gx] + D[gx, gr] - D[gu, gr]
            c2 = D[gr, gx] + D[gx, gv] - D[gr, gv]
            if stale[x]:
                # Una arista nueva bajo la cota es el mínimo exacto (las demás valen >= cota).
                if c1 < best_c[x] or c2 < best_c[x]:
                    if c1 <= c2:
                        best_c[x] = c1
                        best_u[x] = u
                    else:
                        best_c[x] = c2
                        best_u[x] = r
                    stale[x] = False
            else:
                if c1 < best_c[x]:
                    best_c[x] = c1
                    best_u[x] = u
                if c2 < best_c[x]:
                    best_c[x] = c2
                    best_u[x] = r
    path = np.empty(m, np.int64)
    w = s_loc
    for t in range(m):
        path[t] = nodes[w]
        w = nxt[w]
    _open_2opt(path, D, max_iter)
    return path

@njit(cache=True)
def _nn_path(nodes, D, max_iter):
    """Vecino más cercano desde nodes[0] + 2-opt (respaldo cuando no hay anclas válidas)."""
    m = nodes.shape[0]
    used = np.zeros(m, np.bool_)
    path = np.empty(m, np.int64)
    cur = 0
    used[0] = True
    path[0] = nodes[0]
    for t in range(1, m):
        best = -1
        bd = BIG_COST
        for x in range(m):
            if not used[x] and D[nodes[cur], nodes[x]] < bd:
                bd = D[nodes[cur], nodes[x]]
                best = x
        used[best] = True
        path[t] = nodes[best]
        cur = best
    _open_2opt(path, D, max_iter)
    return path

# ---------------------------------------------------------------------------
# Construcción geométrica (índices)
# ---------------------------------------------------------------------------

def subdivide_quadrants_idx(P, xmid, ymid):
    x = P[:, 0]
    y = P[:, 1]
    q1 = (x <= xmid) & (y > ymid)
    q2 = (x > xmid) & (y > ymid)
    q3 = (x <= xmid) & (y <= ymid)
    q4 = ~(q1 | q2 | q3)
    return {
        'Q1': np.nonzero(q1)[0],
        'Q2': np.nonzero(q2)[0],
        'Q3': np.nonzero(q3)[0],
        'Q4': np.nonzero(q4)[0],
    }

def best_frontier_pairs_idx(P, D, A, B, direction, mid, eps, k=FRONTIER_TOPK):
    """Top-k pares (a,b) entre cuadrantes vecinos cerca de la frontera, por distancia."""
    if len(A) == 0 or len(B) == 0:
        return [((None, None), float('inf'))]
    axis = 0 if direction == 'vertical' else 1

    def _candidates(scale):
        band = eps * scale
        return A[np.abs(P[A, axis] - mid) < band], B[np.abs(P[B, axis] - mid) < band]

    candA, candB = _candidates(1)
    scale = 1
    while (len(candA) == 0 or len(candB) == 0) and scale < 16:
        scale *= 2
        candA, candB = _candidates(scale)
    if len(candA) == 0:
        candA = A
    if len(candB) == 0:
        candB = B

    sub = D[np.ix_(candA, candB)].ravel()
    kk = max(1, k)
    order = np.argsort(sub, kind='stable')[:kk]
    nb = len(candB)
    return [((int(candA[o // nb]), int(candB[o % nb])), int(sub[o])) for o in order]

def _quadrant_path(idx, start, end, D, max_iter):
    if len(idx) == 0:
        return np.empty(0, np.int64)
    if len(idx) == 1:
        return idx.astype(np.int64)
    nodes = idx.astype(np.int64)
    if start is None or end is None or start == end:
        return _nn_path(nodes, D, max_iter)
    s_loc = np.nonzero(nodes == start)[0]
    e_loc = np.nonzero(nodes == end)[0]
    if len(s_loc) == 0 or len(e_loc) == 0:
        return _nn_path(nodes, D, max_iter)
    return _anchored_path(nodes, int(s_loc[0]), int(e_loc[0]), D, max_iter)

def _build_assembly_candidate(quads, frontier_choice_map, D, max_iter_local):
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

    connections = []
    for (a, b), _ in frontier_choice_map.values():
        if a is not None and b is not None:
            connections.append((a, b))

    parts = []
    for q in Q_ORDER:
        prev_key, next_key = anchor_map[q]
        parts.append(_quadrant_path(
            quads[q],
            _frontier_node(prev_key, 1),
            _frontier_node(next_key, 0),
            D,
            max(10, min(100, max_iter_local)),
        ))
    return connections, np.concatenate(parts)

# ---------------------------------------------------------------------------
# Pulido
# ---------------------------------------------------------------------------

class _Workspace:
    """Buffers reutilizables por los kernels (evita asignar memoria por kick)."""
    def __init__(self, n):
        self.qbuf = np.zeros(n, np.int64)
        self.inq = np.zeros(n, np.uint8)
        self.journal = np.zeros((JOURNAL_SIZE, 2), np.int64)
        self.jcount = np.zeros(1, np.int64)
        self.best = np.zeros(n, np.int64)
        self.seg = np.zeros(n, np.int64)
        self.ends = np.zeros(8, np.int64)
        self.all_nodes = np.arange(n, dtype=np.int64)

def _positions(tour):
    pos = np.empty(len(tour), np.int64)
    pos[tour] = np.arange(len(tour), dtype=np.int64)
    return pos

def polish_light(tour, D, cand, ncand, ws):
    """Óptimo local completo (2-opt + Or-opt) partiendo de todos los nodos activos."""
    tour = tour.copy()
    n = len(tour)
    if n < 5:
        return tour
    pos = _positions(tour)
    ws.jcount[0] = 0
    _local_search(tour, pos, D, cand, ncand, ws.all_nodes, n, ws.qbuf, ws.inq, ws.journal, ws.jcount, OR_OPT_MAX_L)
    return tour

def polish_ils(tour, D, cand, ncand, ws, deadline):
    """ILS hasta estancamiento (o hasta el tope de tiempo). Devuelve (tour, largo, kicks)."""
    tour = tour.copy()
    n = len(tour)
    pos = _positions(tour)
    cur = int(_tour_len(tour, D))
    if n < 8:
        return tour, cur, 0
    window = max(1, min(max(DB_WINDOW_MIN, n // DB_WINDOW_DIV), (n - 2) // 3))
    limit = stall_limit(n)
    ws.best[:] = tour
    best_len = cur
    temp = ACCEPT_TEMP * cur / n      # fracción del largo promedio de arista
    stall = 0
    kicks = 0
    while stall < limit and time.monotonic() < deadline:
        block = min(ILS_BLOCK, limit - stall)
        cur, best_len, stall = _ils_block(tour, pos, D, cand, ncand, cur, best_len, block, window,
                                          stall, temp, ws.qbuf, ws.inq, ws.journal, ws.jcount,
                                          OR_OPT_MAX_L, ws.best, ws.seg, ws.ends)
        cur = int(cur)
        best_len = int(best_len)
        stall = int(stall)
        kicks += block
    return ws.best.copy(), best_len, kicks

def _cand_array(cand_lists):
    n = len(cand_lists)
    m = max((len(c) for c in cand_lists), default=0)
    cand = np.full((n, max(1, m)), -1, np.int64)
    ncand = np.zeros(n, np.int64)
    for i, c in enumerate(cand_lists):
        cand[i, :len(c)] = c
        ncand[i] = len(c)
    return cand, ncand

def gsph_fc(nodes, MAX_ITER_LOCAL=800, EPS_FRONTIER=5, seed=None):
    """Pipeline GSPH-FC V2. Devuelve (routes, connections, total_len, xmid, ymid, info)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        _seed_kernels(seed)
    n = len(nodes)
    info = {'kicks': 0, 'seed': seed}
    if n == 0:
        return {'QALL': []}, [], 0, 0.0, 0.0, info

    t_start = time.monotonic()
    P = np.asarray(nodes, dtype=float)
    D = build_distance_matrix(nodes)
    knn_k = polish_cand_knn(n)
    cand_m = max(CAND_M, knn_k)
    cand, ncand = _cand_array(build_candidate_sets(nodes, m=cand_m, knn_k=knn_k, use_delaunay=USE_DELAUNAY))
    ws = _Workspace(n)
    profile = get_search_profile(n)

    raw_pool = []
    for part_idx, (part_label, xmid, ymid) in enumerate(generate_partition_portfolio(nodes, limit=profile['partition_limit']), start=1):
        quads = subdivide_quadrants_idx(P, xmid, ymid)
        frontier_options = {}
        for q1, q2, direction in ADJACENCY_CYCLE:
            frontier_options[(q1, q2)] = best_frontier_pairs_idx(
                P, D, quads[q1], quads[q2], direction,
                _frontier_mid(direction, xmid, ymid), EPS_FRONTIER, k=FRONTIER_TOPK,
            )
        partition_candidates = []
        for choice_map in _enumerate_frontier_choice_maps(frontier_options, max_count=profile['assembly_limit']):
            connections, tour = _build_assembly_candidate(quads, choice_map, D, MAX_ITER_LOCAL)
            if len(tour) != n:
                continue
            partition_candidates.append({
                'xmid': xmid, 'ymid': ymid, 'connections': connections,
                'tour': tour, 'raw_len': int(_tour_len(tour, D)),
            })
        partition_candidates.sort(key=lambda item: item['raw_len'])
        raw_pool.extend(partition_candidates[:profile['per_partition_prefilter']])

    if not raw_pool:
        raw_pool.append({'xmid': float(np.median(P[:, 0])), 'ymid': float(np.median(P[:, 1])),
                         'connections': [], 'tour': np.arange(n, dtype=np.int64),
                         'raw_len': int(_tour_len(np.arange(n, dtype=np.int64), D))})
    raw_pool.sort(key=lambda item: item['raw_len'])
    raw_pool = raw_pool[:max(1, profile['light_pool_cap'])]

    pool = []
    for cand_item in raw_pool:
        light = polish_light(cand_item['tour'], D, cand, ncand, ws)
        pool.append(dict(cand_item, tour=light, light_len=int(_tour_len(light, D))))
    pool.sort(key=lambda item: (item['light_len'], item['raw_len']))

    finalists_count = 1
    if len(pool) > 1 and profile['finalists'] > 1:
        if pool[1]['light_len'] <= pool[0]['light_len'] * FINALIST_MARGIN_RATIO:
            finalists_count = min(profile['finalists'], len(pool))

    deadline = t_start + TIME_CAP_S
    best = None
    for finalist in pool[:finalists_count]:
        tour, length, kicks = polish_ils(finalist['tour'], D, cand, ncand, ws, deadline)
        info['kicks'] += kicks
        if best is None or length < best['len']:
            best = {'tour': tour, 'len': length, 'connections': finalist['connections'],
                    'xmid': finalist['xmid'], 'ymid': finalist['ymid']}

    route = [nodes[i] for i in best['tour']]
    total_len = total_tour_length(route)
    if total_len != best['len']:
        raise RuntimeError(f"Longitud inconsistente: ILS={best['len']} recalculada={total_len}")
    conns = [(nodes[a], nodes[b]) for a, b in best['connections']]
    return {'QALL': route}, conns, total_len, best['xmid'], best['ymid'], info

def _warmup():
    """Compila (o carga del caché) los kernels antes de medir tiempos."""
    rng = np.random.RandomState(0)
    pts = [tuple(p) for p in (rng.rand(40, 2) * 1000)]
    saved = os.environ.get('GSPH_STALL')
    os.environ['GSPH_STALL'] = '50'
    try:
        gsph_fc(pts, seed=0)
    finally:
        if saved is None:
            os.environ.pop('GSPH_STALL', None)
        else:
            os.environ['GSPH_STALL'] = saved

def _collect_points(routes_or_route):
    if isinstance(routes_or_route, dict):
        points = []
        for route in routes_or_route.values():
            points.extend(route)
        return points
    return list(routes_or_route)

def _plot_bounds(points, xmid=0.0, ymid=0.0):
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        padding = max((xmax - xmin), (ymax - ymin), 1.0) * 0.08
        return xmin - padding, xmax + padding, ymin - padding, ymax + padding
    return xmid - 100, xmid + 100, ymid - 100, ymid + 100

def _save_figure(fig, save_path):
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=140)
    plt.close(fig)

def _draw_quadrant_background(ax, bounds, xmid, ymid):
    xmin, xmax, ymin, ymax = bounds
    alpha = 0.08
    ax.fill_between([xmin, xmid], [ymid, ymid], [ymax, ymax], color='blue', alpha=alpha)
    ax.fill_between([xmid, xmax], [ymid, ymid], [ymax, ymax], color='green', alpha=alpha)
    ax.fill_between([xmin, xmid], [ymin, ymin], [ymid, ymid], color='red', alpha=alpha)
    ax.fill_between([xmid, xmax], [ymin, ymin], [ymid, ymid], color='orange', alpha=alpha)
    ax.axvline(xmid, linestyle='-', color='black', alpha=0.45, linewidth=1.2)
    ax.axhline(ymid, linestyle='-', color='black', alpha=0.45, linewidth=1.2)
    fontsize = 12
    ax.text(xmin + (xmid-xmin)/2, ymid + (ymax-ymid)/2, 'Q1', ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.65)
    ax.text(xmid + (xmax-xmid)/2, ymid + (ymax-ymid)/2, 'Q2', ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.65)
    ax.text(xmin + (xmid-xmin)/2, ymin + (ymid-ymin)/2, 'Q3', ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.65)
    ax.text(xmid + (xmax-xmid)/2, ymin + (ymid-ymin)/2, 'Q4', ha='center', va='center', fontsize=fontsize, fontweight='bold', alpha=0.65)

def _draw_closed_route(ax, route, color='navy', linewidth=1.8, alpha=0.95, markersize=8):
    if len(route) < 2:
        if len(route) == 1:
            ax.scatter(route[0][0], route[0][1], s=18, color=color, zorder=4)
        return
    x = [p[0] for p in route] + [route[0][0]]
    y = [p[1] for p in route] + [route[0][1]]
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=2)
    ax.scatter([p[0] for p in route], [p[1] for p in route], s=markersize, color=color, alpha=min(1.0, alpha + 0.05), zorder=3)

def _visit_label_step(n):
    if n <= 80:
        return 1
    if n <= 200:
        return 5
    return 15

def plot_final_route(route, save_path, title):
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 10))
    bounds = _plot_bounds(route)
    _draw_closed_route(ax, route, color='#163d7a', linewidth=2.0, alpha=0.95, markersize=10)
    if route:
        ax.scatter(route[0][0], route[0][1], s=90, color='#0f9d58', edgecolors='white', linewidth=1.2, zorder=5, label='Inicio')
        ax.text(route[0][0], route[0][1], '1', color='white', fontsize=8, fontweight='bold', ha='center', va='center', zorder=6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    if route:
        ax.legend(loc='upper right')
    _save_figure(fig, save_path)
    return True

def plot_quadrant_assembly(route, conns, xmid, ymid, save_path, title):
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 10))
    bounds = _plot_bounds(route, xmid, ymid)
    _draw_quadrant_background(ax, bounds, xmid, ymid)
    quads, _, _ = subdivide_quadrants(route, xmid=xmid, ymid=ymid)
    colors = {'Q1': '#2358c9', 'Q2': '#238a3b', 'Q3': '#cc3a3a', 'Q4': '#d97b00'}
    for q in Q_ORDER:
        pts = quads.get(q, [])
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=18, color=colors[q], alpha=0.8, label=q, zorder=3)
    _draw_closed_route(ax, route, color='#222222', linewidth=1.3, alpha=0.55, markersize=0)
    if conns:
        labels = ['Q1-Q2', 'Q2-Q4', 'Q4-Q3', 'Q3-Q1']
        for idx, (a, b) in enumerate(conns):
            ax.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', color='#7b2cbf', linewidth=2.2, alpha=0.95, zorder=4)
            ax.scatter([a[0], b[0]], [a[1], b[1]], color='#7b2cbf', s=48, marker='D', edgecolors='white', linewidth=0.9, zorder=5)
            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            if idx < len(labels):
                ax.text(mx, my, labels[idx], fontsize=8, color='#5a189a', fontweight='bold', ha='center', va='center', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.legend(loc='upper right')
    _save_figure(fig, save_path)
    return True

def plot_visit_order(route, save_path, title):
    if plt is None:
        return False
    if not route:
        return False
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(10, 10))
    bounds = _plot_bounds(route)
    n = len(route)
    closed = route + [route[0]]
    segments = [[closed[i], closed[i+1]] for i in range(n)]
    colors = np.linspace(0, 1, n)
    lc = LineCollection(segments, cmap='viridis', linewidths=2.0)
    lc.set_array(colors)
    ax.add_collection(lc)
    ax.scatter([p[0] for p in route], [p[1] for p in route], c=colors, cmap='viridis', s=16, zorder=3)
    ax.scatter(route[0][0], route[0][1], s=90, color='#0f9d58', edgecolors='white', linewidth=1.2, zorder=5)
    label_step = _visit_label_step(n)
    for idx, p in enumerate(route):
        if idx == 0 or idx == n - 1 or idx % label_step == 0:
            ax.text(p[0], p[1], str(idx + 1), fontsize=7 if n > 120 else 8, color='#111111', ha='left', va='bottom', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.75))
    arrow_step = max(1, label_step)
    for idx in range(0, n, arrow_step):
        a = route[idx]
        b = route[(idx + 1) % n]
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(colors[idx]), lw=1.0, alpha=0.7), zorder=4)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Progreso del recorrido')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    _save_figure(fig, save_path)
    return True

def plot_gsph_fc(routes, conns, xmid, ymid, save_path=None):
    route = routes.get('QALL', []) if isinstance(routes, dict) else list(routes)
    return plot_quadrant_assembly(route, conns, xmid, ymid, save_path, 'GSPH-FC | Ensamblaje')

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
            title += " BEST"
            ax.set_facecolor('#ffffcc')
        
        ax.set_title(title, fontsize=10, fontweight='bold' if is_best else 'normal')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    fig.suptitle(f'Comparación de {n_runs} ejecuciones - {inst_name}', 
                 fontsize=14, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

def _prepare_output_dirs(outdir):
    plots_dir = os.path.join(outdir, 'graficos')
    runs_dir = os.path.join(plots_dir, 'runs')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    for dirpath in (plots_dir, runs_dir):
        for name in os.listdir(dirpath):
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                os.remove(path)

    diag_dir = os.path.join(outdir, 'diagnosticos')
    if os.path.isdir(diag_dir):
        for root, dirs, files in os.walk(diag_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(diag_dir)

    removable_files = {
        'resultados.txt',
        'resumen.txt',
        'multiple_runs_stats.txt',
        'estadisticas_runs.txt',
        'resources.csv',
        'resources.png',
        'events.csv',
        'events_timeline.png',
        'pympler_summary.txt',
        'tracemalloc_top.txt',
        'error.txt',
    }
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if not os.path.isfile(path):
            continue
        if name in removable_files or name.startswith('grafico_'):
            os.remove(path)
    return plots_dir, runs_dir


if __name__ == "__main__":
    import csv
    import math
    try:
        import matplotlib.pyplot as _plt
    except Exception:
        _plt = None
    if plt is None and _plt is not None:
        plt = _plt

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
            print('Usage: python GSPH_V2.py [--profile] [instance_name|file.tsp]')
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

    # Semilla base: GSPH_SEED si está definida; si no, aleatoria (queda anotada en resumen.txt).
    seed_env = os.getenv('GSPH_SEED')
    try:
        base_seed = int(seed_env) if seed_env else None
    except Exception:
        base_seed = None
    if base_seed is None:
        base_seed = int.from_bytes(os.urandom(4), 'little') % (2**31 - 1000)

    _warmup()

    for fname in files:
        inst_name = os.path.splitext(fname)[0]
        outdir = os.path.join(RESULTS_DIR, inst_name)
        os.makedirs(outdir, exist_ok=True)
        plots_dir, runs_dir = _prepare_output_dirs(outdir)
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
            
            run_seed = base_seed + run_idx
            t0 = time.time()

            try:
                routes, conns, total_length, xm, ym, info = gsph_fc(nodes, MAX_ITER_LOCAL, EPS_FRONTIER, seed=run_seed)
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
                'elapsed': elapsed,
                'seed': run_seed,
                'kicks': info['kicks'],
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
                    'best_run_idx': run_idx,
                    'seed': run_seed,
                    'kicks': info['kicks'],
                }
                 
            if num_runs > 1:
                print(f" length={total_length:.2f}, time={elapsed:.3f}s")
        
        if best_result is None:
            continue
            
        routes = best_result['routes']
        conns = best_result['conns'] 
        best_total_length = best_result['total_length']
        xm = best_result['xm']
        ym = best_result['ym']
        elapsed = best_result['elapsed']
        
        if num_runs > 1 and plt is not None:
            best_run_idx = best_result.get('best_run_idx', 0)
            print(f"  Generando {len(all_runs_data)} gráficos...")
            for run_data in all_runs_data:
                idx = run_data['run_idx']
                is_best = (idx == best_run_idx)
                png_name = f"run_{idx+1:02d}{'_best' if is_best else ''}.png"
                png_path_run = os.path.join(runs_dir, png_name)
                title = f"{inst_name} | Run {idx+1}/{num_runs} | L={run_data['total_length']:.2f}"
                if is_best:
                    title += " | MEJOR"
                plot_final_route(run_data['routes'].get('QALL', []), png_path_run, title)
            
            png_comparison = os.path.join(plots_dir, 'comparacion_runs.png')
            _create_comparison_plot(all_runs_data, inst_name, png_comparison, best_run_idx)
        
        route_plot_rel = ''
        assembly_plot_rel = ''
        order_plot_rel = ''
        comparison_plot_rel = ''

        if plt is not None:
            route_plot_rel = 'graficos/ruta_final.png'
            assembly_plot_rel = 'graficos/ensamblaje_cuadrantes.png'
            order_plot_rel = 'graficos/orden_visita.png'
            plot_final_route(
                routes.get('QALL', []),
                os.path.join(outdir, route_plot_rel),
                f'{inst_name} | Ruta final | L={best_total_length:.2f}',
            )
            plot_quadrant_assembly(
                routes.get('QALL', []),
                conns,
                xm,
                ym,
                os.path.join(outdir, assembly_plot_rel),
                f'{inst_name} | Cuadrantes y fronteras',
            )
            plot_visit_order(
                routes.get('QALL', []),
                os.path.join(outdir, order_plot_rel),
                f'{inst_name} | Orden de visita',
            )

        res_txt = os.path.join(outdir, 'resumen.txt')
        with open(res_txt, 'w', encoding='utf-8') as f:
            f.write(f"Instancia: {fname}\n")
            f.write(f"N nodes: {len(nodes)}\n")
            if num_runs > 1:
                f.write(f"Número de ejecuciones: {num_runs}\n")
                f.write(f"MEJOR longitud de ruta: {best_total_length:.2f}\n")
                f.write(f"Longitud promedio     : {sum(all_results)/len(all_results):.2f}\n")
                f.write(f"Rango (min-max)       : {min(all_results):.2f} - {max(all_results):.2f}\n")
            else:
                f.write(f"Longitud total de la ruta: {best_total_length:.2f}\n")
            if plt is not None:
                f.write(f"\n{'='*50}\n")
                f.write("GRÁFICOS PRINCIPALES:\n")
                f.write(f"{'='*50}\n")
                f.write(f"  • {route_plot_rel} - Ruta final cerrada\n")
                f.write(f"  • {assembly_plot_rel} - Cuadrantes, fronteras y elecciones\n")
                f.write(f"  • {order_plot_rel} - Orden de visita de los nodos\n")
                if num_runs > 1:
                    comparison_plot_rel = 'graficos/comparacion_runs.png'
                    f.write(f"  • {comparison_plot_rel} - Comparación visual de runs\n")
                    best_run_idx = best_result.get('best_run_idx', 0)
                    for run_data in all_runs_data:
                        idx = run_data['run_idx']
                        is_best = (idx == best_run_idx)
                        marker = 'MEJOR' if is_best else ''
                        f.write(f"  • graficos/runs/run_{idx+1:02d}{'_best' if is_best else ''}.png - Length: {run_data['total_length']:.2f} {marker}\n")
            else:
                f.write("\nGráficos no disponibles: matplotlib no está cargado.\n")
            f.write(f"Tiempo de ejecución      : {elapsed:.3f} segundos\n")
            f.write(f"Semilla (mejor run)      : {best_result['seed']}\n")
            f.write(f"Kicks ILS (mejor run)    : {best_result['kicks']}\n")
            f.write(f"Numba                    : {'sí' if NUMBA_OK else 'no (Python puro)'}\n")

        notes = 'ok'
        if num_runs > 1:
            stats_file = os.path.join(outdir, 'estadisticas_runs.txt')
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
                writer.writerow([inst_name, len(nodes), f"{best_total_length:.2f}", f"{elapsed:.3f}", route_plot_rel, f'{notes}|avg={avg_length:.2f}|std={std_dev:.2f}'])
            else:
                writer.writerow([inst_name, len(nodes), f"{best_total_length:.2f}", f"{elapsed:.3f}", route_plot_rel, notes])

        if num_runs > 1:
            print(f"Processed {inst_name}: nodes={len(nodes)}, BEST={best_total_length:.2f} (avg={sum(all_results)/len(all_results):.2f}), best_time={elapsed:.3f}s")
        else:
            print(f"Processed {inst_name}: nodes={len(nodes)}, length={best_total_length:.2f}, time={elapsed:.2f}s")

    print('\nAll done. Summary written to', summary_path)

