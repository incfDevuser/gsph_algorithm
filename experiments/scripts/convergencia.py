"""
Curvas de convergencia (GAP del mejor tour vs tiempo) de GSPH_V2 con distintas
construcciones y el MISMO motor de mejora. Responde si la ventaja de la
conectividad de fronteras aparece en los primeros milisegundos y luego se diluye.

Construcciones (las de ablacion_construccion.py): vecino, particion, gsph_fc
(opcionalmente aleatorio y fc). El reloj parte DESPUÉS de construir la matriz de
distancias y las listas de candidatos, que son idénticas para todas; ese tiempo
se guarda aparte. El ILS es el de GSPH_V2 (_ils_block) llamado en bloques chicos
para registrar el mejor largo entre bloques; como el generador aleatorio de numba
conserva su estado entre llamadas, la búsqueda es la misma que con bloques grandes.

Uso:
  python experiments/scripts/convergencia.py --workers 2
  python experiments/scripts/convergencia.py --graficar     # solo rehace figura y tabla
Salida: experiments/results/convergencia/{traza.csv, resumen.csv, convergencia.png}
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
import ablacion_construccion as ab  # noqa: E402

OUT_DIR = os.path.join(EXP_DIR, "results", "convergencia")
INSTANCIAS = ["kroA200", "lin318", "pcb442", "rat783", "pr1002", "fl1577", "d1655", "rl1889"]
CONFIGS = ["vecino", "particion", "gsph_fc"]
BLOQUE = 100          # kicks entre registros
FRACCIONES = [0.01, 0.03, 0.1, 0.3, 1.0]   # puntos de la tabla, en fracción del tiempo medio de GSPH-FC


def ils_trazado(tour, D, cand, ncand, ws, t_ref, deadline, traza, best_global):
    """polish_ils de GSPH_V2 en bloques de BLOQUE kicks, anotando (t, mejor largo global)."""
    tour = tour.copy()
    n = len(tour)
    pos = g._positions(tour)
    cur = int(g._tour_len(tour, D))
    if n < 8:
        return tour, cur
    window = max(1, min(max(g.DB_WINDOW_MIN, n // g.DB_WINDOW_DIV), (n - 2) // 3))
    limit = g.stall_limit(n)
    ws.best[:] = tour
    best_len = cur
    temp = g.ACCEPT_TEMP * cur / n
    stall = 0
    while stall < limit and time.monotonic() < deadline:
        bloque = min(BLOQUE, limit - stall)
        cur, best_len, stall = g._ils_block(tour, pos, D, cand, ncand, cur, best_len, bloque, window,
                                            stall, temp, ws.qbuf, ws.inq, ws.journal, ws.jcount,
                                            g.OR_OPT_MAX_L, ws.best, ws.seg, ws.ends)
        cur, best_len, stall = int(cur), int(best_len), int(stall)
        if best_len < best_global[0]:
            best_global[0] = best_len
            traza.append((time.monotonic() - t_ref, best_len))
    return ws.best.copy(), best_len


def correr(args):
    name, path, n, config, run, seed = args
    nodes = g.read_tsplib(path)
    ab._sembrar(seed)
    t_pre0 = time.monotonic()
    P, D, cand, ncand, ws = ab._preparar(nodes)
    t0 = time.monotonic()
    t_pre = t0 - t_pre0
    traza = []
    best = [1 << 62]

    def anotar(L):
        if L < best[0]:
            best[0] = L
            traza.append((time.monotonic() - t0, L))

    deadline = t0 + g.TIME_CAP_S
    if config == "gsph_fc":
        # Misma lógica que gsph_fc_completo, pero con el ILS trazado.
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
        anotar(raw_pool[0][0])
        pool = []
        for raw_len, tour in raw_pool:
            light = g.polish_light(tour, D, cand, ncand, ws)
            L = int(g._tour_len(light, D))
            anotar(L)
            pool.append((L, raw_len, light))
        pool.sort(key=lambda it: (it[0], it[1]))
        fin = 1
        if len(pool) > 1 and profile['finalists'] > 1 and pool[1][0] <= pool[0][0] * g.FINALIST_MARGIN_RATIO:
            fin = min(profile['finalists'], len(pool))
        for _, _, tour in pool[:fin]:
            ils_trazado(tour, D, cand, ncand, ws, t0, deadline, traza, best)
    else:
        if config == "aleatorio":
            tour = np.random.permutation(n).astype(np.int64)
        elif config == "vecino":
            tour = g._nn_path(np.arange(n, dtype=np.int64), D, 0)
        elif config == "particion":
            tour = ab.construir_particion(P, D)
        else:
            tour = ab.construir_fc(P, D)
        anotar(int(g._tour_len(tour, D)))
        light = g.polish_light(tour, D, cand, ncand, ws)
        anotar(int(g._tour_len(light, D)))
        ils_trazado(light, D, cand, ncand, ws, t0, deadline, traza, best)
    t_fin = time.monotonic() - t0
    return name, n, config, run, seed, t_pre, t_fin, traza


def _curva(traza, malla):
    """Mejor largo vigente en cada instante de la malla (escalón); nan antes del primer tour."""
    ts = np.array([t for t, _ in traza])
    Ls = np.array([L for _, L in traza], dtype=float)
    idx = np.searchsorted(ts, malla, side="right") - 1
    out = np.full(len(malla), np.nan)
    ok = idx >= 0
    out[ok] = Ls[idx[ok]]
    return out


def graficar(p_traza, p_res, p_fig, es=False):
    bks = load_bks()
    corr = defaultdict(list)
    t_fin = defaultdict(list)
    with open(p_traza) as f:
        for r in csv.DictReader(f):
            key = (r["instance"], int(r["n"]), r["config"], int(r["run"]))
            corr[key].append((float(r["t"]), int(r["L"])))
            t_fin[key] = float(r["t_end"])
    insts = sorted({(k[1], k[0]) for k in corr})
    configs = [c for c in ["aleatorio", "vecino", "particion", "fc", "gsph_fc"] if any(k[2] == c for k in corr)]
    etiquetas = {"aleatorio": "Random", "vecino": "Nearest Neighbor", "particion": "Partition (no FC)",
                 "fc": "FC (single)", "gsph_fc": "GSPH-FC"}
    if es:
        etiquetas = {"aleatorio": "Aleatorio", "vecino": "Vecino más cercano",
                     "particion": "Partición (sin FC)", "fc": "FC (una)", "gsph_fc": "GSPH-FC"}
    # Tabla: GAP promedio en fracciones del tiempo medio de GSPH-FC por instancia.
    filas = []
    gap_frac = {c: defaultdict(list) for c in configs}
    for n, inst in insts:
        t_ref = np.mean([t_fin[k] for k in t_fin if k[0] == inst and k[2] == "gsph_fc"])
        for c in configs:
            runs = [k for k in corr if k[0] == inst and k[2] == c]
            for fr in FRACCIONES:
                vals = _curva_all = [100.0 * (_curva(sorted(corr[k]), np.array([fr * t_ref]))[0] - bks[inst]) / bks[inst]
                                     for k in runs]
                gap_frac[c][fr].append(np.nanmean(vals))
                filas.append([inst, n, c, fr, f"{fr * t_ref:.4f}", f"{np.nanmean(vals):.3f}"])
    with open(p_res, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n", "config", "fraction", "t", "gap_avg"])
        w.writerows(filas)
    print("GAP promedio sobre instancias, en fracciones del tiempo medio de GSPH-FC:")
    for c in configs:
        print(f"  {etiquetas[c]:18s} " + "  ".join(f"{fr:>4}: {np.nanmean(gap_frac[c][fr]):7.3f}" for fr in FRACCIONES))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"aleatorio": "#999999", "vecino": "#d62728", "particion": "#ff7f0e",
               "fc": "#9467bd", "gsph_fc": "#1f77b4"}
    ncol = 4
    nfil = int(np.ceil(len(insts) / ncol))
    fig, axes = plt.subplots(nfil, ncol, figsize=(12, 2.7 * nfil), squeeze=False)
    for ax, (n, inst) in zip(axes.ravel(), insts):
        t_max = max(t_fin[k] for k in t_fin if k[0] == inst)
        malla = np.logspace(-3, np.log10(max(t_max, 2e-3)), 300)
        for c in configs:
            runs = [k for k in corr if k[0] == inst and k[2] == c]
            M = np.array([_curva(sorted(corr[k]), malla) for k in runs])
            gap = 100.0 * (M - bks[inst]) / bks[inst]
            with np.errstate(all="ignore"):
                med = np.nanmean(gap, axis=0)
            ax.plot(malla, med, color=colores[c], lw=1.6, label=etiquetas[c])
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{inst} ($n$={n})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    for ax in axes.ravel()[len(insts):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("GAP (%), promedio" if es else "GAP (%), mean of runs", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("tiempo tras el preprocesamiento (s)" if es else "time after preprocessing (s)",
                      fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(p_fig, dpi=200)
    print(f"Figura -> {p_fig}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=INSTANCIAS)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--graficar", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    p_traza = os.path.join(OUT_DIR, "traza.csv")
    p_res = os.path.join(OUT_DIR, "resumen.csv")
    p_fig = os.path.join(OUT_DIR, "convergencia.png")
    if not args.graficar:
        rutas = {name: (n, path) for n, name, path, _ in collect(False)}
        trabajos = [(inst, rutas[inst][1], rutas[inst][0], c, r, args.seed + r)
                    for inst in args.only for c in args.configs for r in range(args.runs)]
        trabajos.sort(key=lambda t: -t[2])
        with open(p_traza, "w", newline="") as f:
            csv.writer(f).writerow(["instance", "n", "config", "run", "seed", "t_pre", "t_end", "t", "L"])
        t_all = time.time()
        with Pool(args.workers, initializer=ab._init_worker) as pool:
            for k, (name, n, c, run, seed, t_pre, t_fin, traza) in enumerate(
                    pool.imap_unordered(correr, trabajos), start=1):
                with open(p_traza, "a", newline="") as f:
                    w = csv.writer(f)
                    for t, L in traza:
                        w.writerow([name, n, c, run, seed, f"{t_pre:.4f}", f"{t_fin:.4f}", f"{t:.6f}", L])
                if k % 10 == 0 or k == len(trabajos):
                    print(f"  {k}/{len(trabajos)} runs ({time.time() - t_all:.0f}s)", flush=True)
    graficar(p_traza, p_res, p_fig)
    graficar(p_traza, p_res, p_fig.replace(".png", "_es.png"), es=True)


if __name__ == "__main__":
    main()
