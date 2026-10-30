"""
Ablación de la construcción de GSPH_V2: ¿cuánto aporta la partición geométrica y
la conectividad de fronteras, con el MISMO motor de mejora?

Cinco construcciones, todas seguidas exactamente del mismo pulido de GSPH_V2
(polish_light = óptimo local 2-opt + Or-opt; polish_ils = ILS con double-bridge
local, aceptación con temperatura y parada por estancamiento), mismas semillas:

  aleatorio  permutación aleatoria (piso: construcción sin información)
  vecino     vecino más cercano desde el nodo 0
  particion  cuadrantes de la caja; cada cuadrante es un camino libre (vecino más
             cercano + 2-opt) y se concatenan en Q1-Q2-Q4-Q3 orientando cada uno
             para empalmar con el anterior. Sin pares de frontera.
  fc         cuadrantes de la caja + el mejor par por frontera (k=1) como anclas de
             los caminos anclados. Sin portafolio ni embudo.
  gsph_fc    pipeline completo de GSPH_V2 (portafolio de particiones, top-3 pares
             por frontera, combinaciones, embudo y finalistas).

Por run se guarda el largo de la construcción, tras la búsqueda local y final,
además de tiempos y kicks. Salida (formato largo + resumen):
  experiments/results/ablacion/ablacion_runs.csv
  experiments/results/ablacion/ablacion_resumen.csv

Uso (desde cualquier carpeta):
  python experiments/scripts/ablacion_construccion.py                 # n <= 6000, set principal
  python experiments/scripts/ablacion_construccion.py --only kroE100 pcb442 --runs 3
  python experiments/scripts/ablacion_construccion.py --workers 3
  python experiments/scripts/ablacion_construccion.py --resumen       # solo re-agrega el CSV
"""
import argparse, csv, os, random, sys, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(EXP_DIR, "algorithms"))
sys.path.insert(0, SCRIPT_DIR)
import GSPH_V2 as g  # noqa: E402
from benchmark_v2 import load_bks, collect  # noqa: E402

OUT_DIR = os.path.join(EXP_DIR, "results", "ablacion")
CONFIGS = ["aleatorio", "vecino", "particion", "fc", "gsph_fc"]
EXCLUIDAS = {"eil10", "eil20", "linhp318"}   # sin BKS TSPLIB / BKS de otra variante
COLS = ["instance", "n", "config", "run", "seed", "L_raw", "L_light", "L_final",
        "t_constr", "t_total", "kicks", "finalists"]


def _sembrar(seed):
    random.seed(seed)
    np.random.seed(seed)
    g._seed_kernels(seed)


def _preparar(nodes):
    n = len(nodes)
    D = g.build_distance_matrix(nodes)
    knn_k = g.polish_cand_knn(n)
    cand, ncand = g._cand_array(g.build_candidate_sets(nodes, m=max(g.CAND_M, knn_k), knn_k=knn_k,
                                                       use_delaunay=g.USE_DELAUNAY))
    return np.asarray(nodes, dtype=float), D, cand, ncand, g._Workspace(n)


def _caja(P):
    return (P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 1].min() + P[:, 1].max()) / 2


def _iter_local():
    return max(10, min(100, g.MAX_ITER_LOCAL))


def construir_particion(P, D):
    """Cuadrantes sin conectividad de fronteras: caminos libres orientados para empalmar."""
    quads = g.subdivide_quadrants_idx(P, *_caja(P))
    partes = []
    for q in g.Q_ORDER:
        idx = quads[q].astype(np.int64)
        if len(idx) == 0:
            continue
        path = g._nn_path(idx, D, _iter_local()) if len(idx) > 1 else idx
        if partes:
            fin = partes[-1][-1]
            if D[fin, path[-1]] < D[fin, path[0]]:
                path = path[::-1].copy()
        partes.append(path)
    return np.concatenate(partes)


def construir_fc(P, D):
    """Cuadrantes de la caja + mejor par por frontera (k=1) como anclas."""
    xmid, ymid = _caja(P)
    quads = g.subdivide_quadrants_idx(P, xmid, ymid)
    opciones = {}
    for q1, q2, direction in g.ADJACENCY_CYCLE:
        opciones[(q1, q2)] = g.best_frontier_pairs_idx(P, D, quads[q1], quads[q2], direction,
                                                      g._frontier_mid(direction, xmid, ymid),
                                                      g.EPS_FRONTIER, k=1)
    choice = g._enumerate_frontier_choice_maps(opciones, max_count=1)[0]
    return g._build_assembly_candidate(quads, choice, D, g.MAX_ITER_LOCAL)[1]


def gsph_fc_completo(nodes, P, D, cand, ncand, ws, t_start):
    """Réplica de GSPH_V2.gsph_fc que además registra los largos intermedios."""
    n = len(nodes)
    profile = g.get_search_profile(n)
    raw_pool = []
    for _, xmid, ymid in g.generate_partition_portfolio(nodes, limit=profile['partition_limit']):
        quads = g.subdivide_quadrants_idx(P, xmid, ymid)
        opciones = {}
        for q1, q2, direction in g.ADJACENCY_CYCLE:
            opciones[(q1, q2)] = g.best_frontier_pairs_idx(P, D, quads[q1], quads[q2], direction,
                                                          g._frontier_mid(direction, xmid, ymid),
                                                          g.EPS_FRONTIER, k=g.FRONTIER_TOPK)
        cands = []
        for choice in g._enumerate_frontier_choice_maps(opciones, max_count=profile['assembly_limit']):
            tour = g._build_assembly_candidate(quads, choice, D, g.MAX_ITER_LOCAL)[1]
            if len(tour) == n:
                cands.append((int(g._tour_len(tour, D)), tour))
        cands.sort(key=lambda it: it[0])
        raw_pool.extend(cands[:profile['per_partition_prefilter']])
    raw_pool.sort(key=lambda it: it[0])
    raw_pool = raw_pool[:max(1, profile['light_pool_cap'])]
    t_constr = time.monotonic() - t_start
    pool = []
    for raw_len, tour in raw_pool:
        light = g.polish_light(tour, D, cand, ncand, ws)
        pool.append((int(g._tour_len(light, D)), raw_len, light))
    pool.sort(key=lambda it: (it[0], it[1]))
    fin = 1
    if len(pool) > 1 and profile['finalists'] > 1 and pool[1][0] <= pool[0][0] * g.FINALIST_MARGIN_RATIO:
        fin = min(profile['finalists'], len(pool))
    deadline = t_start + g.TIME_CAP_S
    best, kicks = None, 0
    for light_len, raw_len, tour in pool[:fin]:
        t, L, k = g.polish_ils(tour, D, cand, ncand, ws, deadline)
        kicks += k
        if best is None or L < best:
            best = L
    # El largo "raw" y "light" reportados son los del mejor candidato tras la búsqueda local.
    return raw_pool[0][0], pool[0][0], best, t_constr, kicks, fin


def correr(args):
    name, path, n, config, run, seed = args
    nodes = g.read_tsplib(path)
    _sembrar(seed)
    t0 = time.monotonic()
    P, D, cand, ncand, ws = _preparar(nodes)
    if config == "gsph_fc":
        L_raw, L_light, L_final, t_constr, kicks, fin = gsph_fc_completo(nodes, P, D, cand, ncand, ws, t0)
    else:
        if config == "aleatorio":
            tour = np.random.permutation(n).astype(np.int64)
        elif config == "vecino":
            tour = g._nn_path(np.arange(n, dtype=np.int64), D, 0)
        elif config == "particion":
            tour = construir_particion(P, D)
        else:
            tour = construir_fc(P, D)
        if sorted(tour.tolist()) != list(range(n)):
            raise RuntimeError(f"{name}/{config}: construcción inválida")
        t_constr = time.monotonic() - t0
        L_raw = int(g._tour_len(tour, D))
        light = g.polish_light(tour, D, cand, ncand, ws)
        L_light = int(g._tour_len(light, D))
        final, L_final, kicks = g.polish_ils(light, D, cand, ncand, ws, t0 + g.TIME_CAP_S)
        if sorted(final.tolist()) != list(range(n)) or int(g._tour_len(final, D)) != L_final:
            raise RuntimeError(f"{name}/{config}: tour final inválido")
        fin = 1
    return [name, n, config, run, seed, L_raw, L_light, int(L_final),
            f"{t_constr:.3f}", f"{time.monotonic() - t0:.3f}", int(kicks), fin]


def _init_worker():
    g._warmup()


def resumir(path_runs, path_out):
    bks = load_bks()
    grupos = defaultdict(list)
    with open(path_runs) as f:
        for r in csv.DictReader(f):
            grupos[(r["instance"], int(r["n"]), r["config"])].append(r)
    filas = []
    for (inst, n, config), rs in sorted(grupos.items(), key=lambda kv: (kv[0][1], kv[0][0], CONFIGS.index(kv[0][2]))):
        b = bks[inst]
        gap = lambda v: 100.0 * (v - b) / b
        fin = [int(r["L_final"]) for r in rs]
        filas.append([inst, n, config, len(rs),
                      f"{np.mean([gap(int(r['L_raw'])) for r in rs]):.3f}",
                      f"{np.mean([gap(int(r['L_light'])) for r in rs]):.3f}",
                      f"{gap(min(fin)):.3f}", f"{np.mean([gap(v) for v in fin]):.3f}",
                      sum(v <= b for v in fin),
                      f"{np.mean([float(r['t_constr']) for r in rs]):.3f}",
                      f"{np.mean([float(r['t_total']) for r in rs]):.3f}",
                      int(np.mean([int(r['kicks']) for r in rs]))])
    with open(path_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n", "config", "runs", "gap_raw_avg", "gap_light_avg", "gap_best",
                    "gap_avg", "hits", "t_constr_avg", "t_total_avg", "kicks_avg"])
        w.writerows(filas)
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10, help="runs por instancia con n <= 2000")
    ap.add_argument("--runs-grandes", type=int, default=3, help="runs por instancia con n > 2000")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-n", type=int, default=6000, help="solo instancias con n <= este valor")
    ap.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resumen", action="store_true", help="solo re-agregar ablacion_runs.csv")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    p_runs = os.path.join(OUT_DIR, "ablacion_runs.csv")
    p_res = os.path.join(OUT_DIR, "ablacion_resumen.csv")
    if args.resumen:
        resumir(p_runs, p_res)
        print(f"Resumen -> {p_res}")
        return

    bks = load_bks()
    items = [it for it in collect(False) if it[1] not in EXCLUIDAS and it[1] in bks and it[0] <= args.max_n]
    if args.only:
        items = [it for it in items if it[1] in set(args.only)]
    trabajos = []
    for n, name, path, _ in items:
        runs = args.runs if n <= 2000 else args.runs_grandes
        for config in args.configs:
            for r in range(runs):
                trabajos.append((name, path, n, config, r, args.seed + r))
    # Las instancias grandes primero, para que el pool no termine esperando una sola.
    trabajos.sort(key=lambda t: -t[2])
    print(f"{len(items)} instancias, {len(trabajos)} runs, {args.workers} procesos", flush=True)
    with open(p_runs, "w", newline="") as f:
        csv.writer(f).writerow(COLS)
    t_all = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, fila in enumerate(pool.imap_unordered(correr, trabajos), start=1):
            with open(p_runs, "a", newline="") as f:
                csv.writer(f).writerow(fila)
            if k % 50 == 0 or k == len(trabajos):
                print(f"  {k}/{len(trabajos)} runs ({time.time() - t_all:.0f}s)", flush=True)
    resumir(p_runs, p_res)
    print(f"Total {time.time() - t_all:.0f}s -> {p_runs}", flush=True)


if __name__ == "__main__":
    main()
