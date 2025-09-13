import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
from scipy.spatial import cKDTree 

EPS_FRONTIER      = 5
MAX_ITER_LOCAL    = 800
RESULTS_DIR       = "gsph_fc_results"
BASE_DIR          = "./src/INSTANCES"

TIME_BUDGET_S     = 1.5 
CAND_M            = 12    
CAND_KNN          = 16   
USE_DELAUNAY      = False 
MAX_SWEEPS_2OPT   = 1  
OR_OPT_MAX_L      = 1  
OR_OPT_MOVE_LIMIT = 3000 
ILS_ITERS         = 1     
SKIP_PROB_2OPT    = 0.35 

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

def subdivide_quadrants(pts):
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

def best_frontier_pair(A, B, direction, mid, eps):
    filt = (lambda p: abs(p[0]-mid)<eps) if direction=='vertical' else (lambda p: abs(p[1]-mid)<eps)
    candA = [p for p in A if filt(p)]
    candB = [p for p in B if filt(p)]
    best, best_d = (None,None), float('inf')
    for a in candA:
        for b in candB:
            d = euclidean_distance(a,b)
            if d < best_d:
                best, best_d = (a,b), d
    return best, best_d

def tsp_2opt(points, max_iter=50):
    """
    2-opt simple para PATH abierto usando euclidean_distance.
    Devuelve la ruta mejorada.
    """
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
                        best_gain = gain
                        best_c = c
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

def or_opt_move(perm, D, cand, Lmax=OR_OPT_MAX_L, move_limit=OR_OPT_MOVE_LIMIT, deadline=None):
    """Mueve bloques cortos en tour circular. L=1 por defecto para máxima velocidad."""
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

def polish_fast(nodes, route):
    """Pulido con presupuesto de tiempo: 2-opt(DLB) → Or-opt → 2-opt(DLB) → ILS."""
    if len(route) < 10:
        return route[:]
    D = build_distance_matrix(nodes)
    cand = build_candidate_sets(nodes, m=CAND_M, knn_k=CAND_KNN, use_delaunay=USE_DELAUNAY)

    deadline = time.monotonic() + max(0.05, TIME_BUDGET_S)
    raw_perm = route_to_perm(route, nodes)
    perm = ensure_valid_perm(raw_perm, len(nodes), D)
    perm = two_opt_dlb(perm, D, cand, max_sweeps=MAX_SWEEPS_2OPT, deadline=deadline)
    perm = or_opt_move(perm, D, cand, Lmax=OR_OPT_MAX_L, deadline=deadline)
    perm = two_opt_dlb(perm, D, cand, max_sweeps=1, deadline=deadline)

    best = perm[:]; best_len = tour_len_from_perm(best, D)
    for _ in range(max(0, ILS_ITERS)):
        if time.monotonic() > deadline: break
        kick = double_bridge_on_perm(best)
        kick = ensure_valid_perm(kick, len(nodes), D)
        kick = two_opt_dlb(kick, D, cand, max_sweeps=1, deadline=deadline)
        L = tour_len_from_perm(kick, D)
        if L < best_len:
            best, best_len = kick[:], L

    return perm_to_route(best, nodes)
def _concat_routes_q_order(routes):
    order = ['Q1','Q2','Q4','Q3']
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
def gsph_fc(nodes, MAX_ITER_LOCAL=800 ,EPS_FRONTIER=5):
    quads, xmid, ymid = subdivide_quadrants(nodes)
    routes = {}
    for q, pts in quads.items():
        if len(pts) > 1:
            start = pts[0]
            remaining = set(pts); remaining.remove(start)
            curr = start; seed = [curr]
            while remaining:
                nxt = min(remaining, key=lambda p: eucl_float(curr, p))
                seed.append(nxt); remaining.remove(nxt); curr = nxt
            rt = seed
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
    connections = []
    inter_len   = 0
    for q1,q2,dirc,mid in neighbor_pairs:
        (a,b), d = best_frontier_pair(quads[q1], quads[q2], dirc, mid, EPS_FRONTIER)
        if a and b:
            connections.append((a,b))
            inter_len += d
    initial_tour = _concat_routes_q_order(routes)
    improved_tour = polish_fast(nodes, initial_tour)
    total_len = total_tour_length(improved_tour)

    routes_out = {'QALL': improved_tour}
    return routes_out, connections, total_len, xmid, ymid

def plot_gsph_fc(routes, conns, xmid, ymid, save_path=None):
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
def total_path_length_float(path):
    if len(path) < 2: return 0
    return sum(eucl_float(path[i], path[i+1]) for i in range(len(path)-1))

def total_tour_length_float(tour):
    if len(tour) < 2: return 0
    return total_path_length_float(tour) + eucl_float(tour[-1], tour[0])

def optimize_route(depot, orders, method="gsph"):
    """
    Entrada: 
      - depot: {id, name, lat, lng}
      - orders: [{id, lat, lng}, ...]
      - method: "gsph" (original) o "genetic" (algoritmo genético)

    Salida:
      - optimized_coords: [[lat, lng], ...]
      - total_length: float
    """
    if method == "genetic":
        # Importar el módulo genético
        try:
            from gsph_genetic import optimize_route_genetic
            return optimize_route_genetic(depot, orders)
        except ImportError:
            print("Módulo genético no disponible, usando GSPH original")
            method = "gsph"
    
    # Método GSPH original
    nodes = [(depot["lat"], depot["lng"])] + [(o["lat"], o["lng"]) for o in orders]
    routes, _, _, _, _ = gsph_fc(nodes)
    optimized_coords = [[float(lat), float(lng)] for lat, lng in routes['QALL']]

    total_length = total_tour_length_float(routes['QALL'])

    return optimized_coords, total_length


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tsp_path = os.path.join(BASE_DIR, 'eil101.tsp')
    nodes = read_tsplib(tsp_path)
    t0 = time.time()
    routes, conns, total_length, xm, ym = gsph_fc(nodes, MAX_ITER_LOCAL, EPS_FRONTIER)
    t_heur = time.time() - t0

    with open(os.path.join(RESULTS_DIR, "resultados.txt"), "w", encoding="utf-8") as f:
        f.write("── RESULTADOS GSPH–FC (RÁPIDO) ──\n")
        f.write(f"Instancia: {os.path.basename(tsp_path)}\n")
        f.write(f"Longitud total de la ruta: {total_length:.2f}\n")
        f.write(f"Tiempo de ejecución      : {t_heur:.3f} segundos\n")
        f.write(f"Time budget (pulido)     : {TIME_BUDGET_S:.2f} s\n")
        f.write(f"cand_m={CAND_M}, knn_k={CAND_KNN}, 2opt_sweeps={MAX_SWEEPS_2OPT}, "
                f"orL={OR_OPT_MAX_L}, ILS={ILS_ITERS}\n")

    print("\n── RESULTADOS GSPH–FC (RÁPIDO) ──")
    print(f"Instancia: {os.path.basename(tsp_path)}")
    print(f"Longitud total de la ruta: {total_length:.2f}")
    print(f"Tiempo de ejecución      : {t_heur:.3f} segundos")
    print(f"(pulido con budget ~{TIME_BUDGET_S}s)")

    plot_gsph_fc(routes, conns, xm, ym, save_path=os.path.join(RESULTS_DIR, "grafico_gsph_fc.png"))
