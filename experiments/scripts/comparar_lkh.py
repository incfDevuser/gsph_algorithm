"""
Comparación de GSPH_V2 contra LKH (Helsgaun) en las mismas instancias y semillas.

LKH se compila desde la fuente oficial en experiments/data/lkh/ (no versionado):
  cd experiments/data/lkh && tar xzf LKH-2.0.10.tgz && make -C LKH-2.0.10
Si ese binario no existe se usa el LKH-3 embebido en el paquete elkai (pip install elkai;
en Python 3.14 requiere PIP_CONSTRAINT con "scikit-build-core<0.10" para compilar),
pasándole los mismos parámetros.

Dos modos de LKH, cada run es una ejecución independiente con RUNS = 1 y SEED = semilla:
  default    parámetros por defecto de LKH (MAX_TRIALS = n, candidatos alfa).
  mismo_t    TIME_LIMIT = tiempo medio por run de GSPH-FC en esa instancia
             (leído de experiments/results/v2/benchmark_v2.csv). LKH no cuenta el
             preprocesamiento (ascenso) dentro de TIME_LIMIT, así que se reporta el
             tiempo total medido desde afuera.
Cada tour de LKH se lee y su largo se recalcula con la métrica EUC_2D del repo.

Uso:
  python experiments/scripts/comparar_lkh.py --workers 3
  python experiments/scripts/comparar_lkh.py --only kroE100 pr1002 --runs 2
  python experiments/scripts/comparar_lkh.py --resumen
Salida: experiments/results/lkh/{lkh_runs.csv, lkh_resumen.csv}
"""
import argparse, csv, os, re, subprocess, sys, tempfile, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(EXP_DIR, "algorithms"))
sys.path.insert(0, SCRIPT_DIR)
import GSPH_V2 as g  # noqa: E402
from benchmark_v2 import load_bks, collect  # noqa: E402

LKH_BIN = os.getenv("LKH_BIN", os.path.join(EXP_DIR, "data", "lkh", "LKH-2.0.10", "LKH"))
V2_CSV = os.path.join(EXP_DIR, "results", "v2", "benchmark_v2.csv")
OUT_DIR = os.path.join(EXP_DIR, "results", "lkh")
EXCLUIDAS = {"eil10", "eil20", "linhp318"}
MODOS = ["default", "mismo_t"]
TOPE_S = 3600   # tope de seguridad por run


def tiempos_gsph():
    with open(V2_CSV) as f:
        return {r["instance"]: float(r["t_avg"]) for r in csv.DictReader(f)}


def leer_tour(path):
    tour, leyendo = [], False
    with open(path) as f:
        for linea in f:
            linea = linea.strip()
            if linea == "TOUR_SECTION":
                leyendo = True
                continue
            if leyendo:
                v = int(linea)
                if v == -1:
                    break
                tour.append(v - 1)
    return tour


def _correr_elkai(path, modo, seed, t_lim):
    """LKH-3 embebido en elkai: mismos parámetros que el binario, problema EUC_2D por stdin."""
    from elkai import _elkai
    nodes = g.read_tsplib(path)
    params = f"RUNS = 1\nSEED = {seed}\nTRACE_LEVEL = 0\nPROBLEM_FILE = :stdin:\n"
    if modo == "mismo_t":
        params += f"TIME_LIMIT = {t_lim:.3f}\n"
    problem = (f"TYPE : TSP\nDIMENSION : {len(nodes)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
               + "".join(f"{i + 1} {x:.10g} {y:.10g}\n" for i, (x, y) in enumerate(nodes)) + "EOF\n")
    t0 = time.monotonic()
    sol = _elkai.solve_problem(params, problem)
    t = time.monotonic() - t0
    return [v - 1 for v in sol], t, float("nan")


def correr(args):
    name, path, n, modo, run, seed, t_lim = args
    if not os.path.exists(LKH_BIN):
        tour, t, t_pre = _correr_elkai(path, modo, seed, t_lim)
        nodes = g.read_tsplib(path)
        if sorted(tour) != list(range(len(nodes))):
            raise RuntimeError(f"{name}/{modo}: tour de LKH inválido")
        L = g.total_tour_length([nodes[i] for i in tour])
        return [name, n, modo, run, seed, L, f"{t:.3f}", f"{t_pre:.3f}", f"{t_lim:.3f}"]
    with tempfile.TemporaryDirectory() as tmp:
        par = os.path.join(tmp, "p.par")
        tour_f = os.path.join(tmp, "out.tour")
        lineas = [f"PROBLEM_FILE = {os.path.abspath(path)}", "RUNS = 1", f"SEED = {seed}",
                  f"OUTPUT_TOUR_FILE = {tour_f}", "TRACE_LEVEL = 1"]
        if modo == "mismo_t":
            lineas.append(f"TIME_LIMIT = {t_lim:.3f}")
        with open(par, "w") as f:
            f.write("\n".join(lineas) + "\n")
        t0 = time.monotonic()
        out = subprocess.run([LKH_BIN, par], capture_output=True, text=True, timeout=TOPE_S)
        t = time.monotonic() - t0
        if out.returncode != 0 or not os.path.exists(tour_f):
            raise RuntimeError(f"{name}/{modo}: LKH falló\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        m = re.search(r"Preprocessing time = ([\d.]+) sec", out.stdout)
        t_pre = float(m.group(1)) if m else float("nan")
        tour = leer_tour(tour_f)
    nodes = g.read_tsplib(path)
    if sorted(tour) != list(range(len(nodes))):
        raise RuntimeError(f"{name}/{modo}: tour de LKH inválido")
    L = g.total_tour_length([nodes[i] for i in tour])
    return [name, n, modo, run, seed, L, f"{t:.3f}", f"{t_pre:.3f}", f"{t_lim:.3f}"]


def resumir(p_runs, p_out):
    bks = load_bks()
    v2 = {}
    with open(V2_CSV) as f:
        for r in csv.DictReader(f):
            v2[r["instance"]] = r
    grupos = defaultdict(list)
    with open(p_runs) as f:
        for r in csv.DictReader(f):
            grupos[(r["instance"], int(r["n"]), r["mode"])].append(r)
    filas = []
    for (inst, n, modo), rs in sorted(grupos.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        b = bks[inst]
        L = [int(r["L"]) for r in rs]
        filas.append([inst, n, modo, len(rs), f"{100 * (min(L) - b) / b:.3f}",
                      f"{100 * (np.mean(L) - b) / b:.3f}", sum(x <= b for x in L),
                      f"{np.mean([float(r['t']) for r in rs]):.2f}",
                      f"{np.mean([float(r['t_pre']) for r in rs]):.2f}",
                      v2[inst]["gap_best"], v2[inst]["gap_avg"], v2[inst]["hits"], v2[inst]["t_avg"]])
    with open(p_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n", "mode", "runs", "lkh_gap_best", "lkh_gap_avg", "lkh_hits", "lkh_t",
                    "lkh_t_pre", "gsph_gap_best", "gsph_gap_avg", "gsph_hits", "gsph_t"])
        w.writerows(filas)
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10, help="runs por instancia con n <= 2000")
    ap.add_argument("--runs-grandes", type=int, default=3, help="runs por instancia con n > 2000")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--modos", nargs="*", default=MODOS, choices=MODOS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resumen", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    p_runs = os.path.join(OUT_DIR, "lkh_runs.csv")
    p_res = os.path.join(OUT_DIR, "lkh_resumen.csv")
    if not args.resumen:
        if not os.path.exists(LKH_BIN):
            try:
                from elkai import _elkai  # noqa: F401
                print(f"Sin binario en {LKH_BIN}: se usa LKH-3 embebido en elkai", flush=True)
            except ImportError:
                sys.exit(f"No está el binario de LKH en {LKH_BIN} ni elkai (ver la cabecera)")
        bks, t_g = load_bks(), tiempos_gsph()
        items = [it for it in collect(False) if it[1] not in EXCLUIDAS and it[1] in bks]
        if args.only:
            items = [it for it in items if it[1] in set(args.only)]
        trabajos = []
        for n, name, path, _ in items:
            runs = args.runs if n <= 2000 else args.runs_grandes
            for modo in args.modos:
                for r in range(runs):
                    trabajos.append((name, path, n, modo, r, args.seed + r, t_g[name]))
        trabajos.sort(key=lambda t: -t[2])
        with open(p_runs, "w", newline="") as f:
            csv.writer(f).writerow(["instance", "n", "mode", "run", "seed", "L", "t", "t_pre", "t_limit"])
        t_all = time.time()
        with Pool(args.workers) as pool:
            for k, fila in enumerate(pool.imap_unordered(correr, trabajos), start=1):
                with open(p_runs, "a", newline="") as f:
                    csv.writer(f).writerow(fila)
                if k % 50 == 0 or k == len(trabajos):
                    print(f"  {k}/{len(trabajos)} runs ({time.time() - t_all:.0f}s)", flush=True)
    resumir(p_runs, p_res)
    print(f"Resumen -> {p_res}")


if __name__ == "__main__":
    main()
