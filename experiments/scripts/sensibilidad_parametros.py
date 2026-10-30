"""
Sensibilidad de los parámetros de GSPH_V2, uno a la vez alrededor de los valores
por defecto, sobre el subconjunto de ajuste (kroE100 lin318 pcb442 rat783 pr1002 d1655).

Parámetros (constante del módulo GSPH_V2 -> valores probados; el default va en *):
  FRONTIER_TOPK              pares por frontera           1, 3*, 5
  PARTITION_PORTFOLIO_LIMIT  particiones del portafolio   1, 6* (efectivo 3-5 según n)
  ACCEPT_TEMP                temperatura / largo medio     0, 0.1, 0.2*, 0.4
  DB_WINDOW_DIV              ventana W(n) = max(50, n/DIV) 4, 8*, 16
  STALL_FACTOR               parada S(n) = max(5000, F*n)  100, 300*, 1000

Uso:
  python experiments/scripts/sensibilidad_parametros.py --workers 3
  python experiments/scripts/sensibilidad_parametros.py --resumen
Salida: experiments/results/sensibilidad/{sens_runs.csv, sens_resumen.csv}
"""
import argparse, csv, os, sys, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(EXP_DIR, "algorithms"))
sys.path.insert(0, SCRIPT_DIR)
import GSPH_V2 as g  # noqa: E402
from benchmark_v2 import load_bks, collect  # noqa: E402

OUT_DIR = os.path.join(EXP_DIR, "results", "sensibilidad")
INSTANCIAS = ["kroE100", "lin318", "pcb442", "rat783", "pr1002", "d1655"]
DEFAULTS = {"FRONTIER_TOPK": 3, "PARTITION_PORTFOLIO_LIMIT": 6, "ACCEPT_TEMP": 0.2,
            "DB_WINDOW_DIV": 8, "STALL_FACTOR": 300}
VARIANTES = [("default", None, None)] + [
    (p, p, v) for p, vals in [
        ("FRONTIER_TOPK", [1, 5]),
        ("PARTITION_PORTFOLIO_LIMIT", [1]),
        ("ACCEPT_TEMP", [0.0, 0.1, 0.4]),
        ("DB_WINDOW_DIV", [4, 16]),
        ("STALL_FACTOR", [100, 1000]),
    ] for v in vals
]


def _etiqueta(param, valor):
    return "default" if param is None else f"{param}={valor}"


def correr(args):
    name, path, param, valor, run, seed = args
    for p, v in DEFAULTS.items():
        setattr(g, p, v)
    if param is not None:
        setattr(g, param, valor)
    nodes = g.read_tsplib(path)
    t0 = time.monotonic()
    routes, _, L, _, _, info = g.gsph_fc(nodes, seed=seed)
    if sorted(routes["QALL"]) != sorted(nodes):
        raise RuntimeError(f"{name}: tour inválido")
    return [name, len(nodes), _etiqueta(param, valor), run, seed, L,
            f"{time.monotonic() - t0:.3f}", info["kicks"]]


def resumir(p_runs, p_out):
    bks = load_bks()
    grupos = defaultdict(list)
    with open(p_runs) as f:
        for r in csv.DictReader(f):
            grupos[(r["variant"], r["instance"])].append(r)
    orden = [_etiqueta(p, v) for _, p, v in VARIANTES]
    filas = []
    for var in orden:
        gb, ga, tt, hits, runs = [], [], [], 0, 0
        for inst in INSTANCIAS:
            rs = grupos.get((var, inst), [])
            if not rs:
                continue
            b = bks[inst]
            L = [int(r["L"]) for r in rs]
            gb.append(100.0 * (min(L) - b) / b)
            ga.append(100.0 * (np.mean(L) - b) / b)
            tt.append(np.mean([float(r["t"]) for r in rs]))
            hits += sum(x <= b for x in L)
            runs += len(L)
        if gb:
            filas.append([var, f"{np.mean(gb):.3f}", f"{np.mean(ga):.3f}", f"{hits}/{runs}",
                          f"{np.mean(tt):.2f}"])
    with open(p_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "gap_best_avg", "gap_avg_avg", "hits", "t_avg"])
        w.writerows(filas)
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resumen", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    p_runs = os.path.join(OUT_DIR, "sens_runs.csv")
    p_res = os.path.join(OUT_DIR, "sens_resumen.csv")
    if not args.resumen:
        rutas = {name: path for _, name, path, _ in collect(False)}
        trabajos = [(inst, rutas[inst], p, v, r, args.seed + r)
                    for _, p, v in VARIANTES for inst in INSTANCIAS for r in range(args.runs)]
        trabajos.sort(key=lambda t: -INSTANCIAS.index(t[0]))
        with open(p_runs, "w", newline="") as f:
            csv.writer(f).writerow(["instance", "n", "variant", "run", "seed", "L", "t", "kicks"])
        t_all = time.time()
        with Pool(args.workers, initializer=g._warmup) as pool:
            for k, fila in enumerate(pool.imap_unordered(correr, trabajos), start=1):
                with open(p_runs, "a", newline="") as f:
                    csv.writer(f).writerow(fila)
                if k % 50 == 0 or k == len(trabajos):
                    print(f"  {k}/{len(trabajos)} runs ({time.time() - t_all:.0f}s)", flush=True)
    for fila in resumir(p_runs, p_res):
        print(" ".join(fila))


if __name__ == "__main__":
    main()
