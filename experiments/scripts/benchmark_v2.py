"""
Benchmark rápido de GSPH_V2 (sin gráficos) sobre instances/standard y las
instancias EUC_2D de instances/large. Pensado para iterar sobre el algoritmo:
valida cada tour, calcula GAP contra benchmarks/BKS/BKS_TSP.txt y escribe un CSV.

Uso:
  python benchmark_v2.py                      # todas, 10 runs (standard) / 3 runs (large)
  python benchmark_v2.py --runs 5 --large-runs 1
  python benchmark_v2.py --only pcb442 pr1002 # subconjunto
  python benchmark_v2.py --no-large
"""
import argparse, csv, os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(EXP_DIR, "algorithms"))
import GSPH_V2 as g  # noqa: E402

STANDARD_DIR = os.path.join(EXP_DIR, "instances", "standard")
LARGE_DIR = os.path.join(EXP_DIR, "instances", "large")
BKS_FILE = os.path.join(EXP_DIR, "benchmarks", "BKS", "BKS_TSP.txt")
OUT_CSV = os.path.join(EXP_DIR, "results", "v2", "benchmark_v2.csv")


def load_bks():
    bks = {}
    with open(BKS_FILE) as f:
        for line in f:
            if ":" in line:
                name, val = line.split(":", 1)
                try:
                    bks[name.strip()] = int(val.strip())
                except ValueError:
                    pass
    return bks


def header(path):
    """Lee DIMENSION y EDGE_WEIGHT_TYPE de la cabecera TSPLIB."""
    info = {}
    with open(path) as f:
        for line in f:
            if line.strip() == "NODE_COORD_SECTION":
                break
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip().upper()] = v.strip()
    return int(info.get("DIMENSION", 0)), info.get("EDGE_WEIGHT_TYPE", "")


def collect(include_large):
    items = []
    for d, kind in ((STANDARD_DIR, "standard"), (LARGE_DIR, "large")):
        if kind == "large" and not include_large:
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".tsp"):
                continue
            path = os.path.join(d, fname)
            n, ewt = header(path)
            if ewt != "EUC_2D":   # el solver solo calcula EUC_2D; otras métricas darían GAP sin sentido
                continue
            items.append((n, fname[:-4], path, kind))
    items.sort()
    return items


def main():
    ap = argparse.ArgumentParser(description="Benchmark rápido de GSPH_V2")
    ap.add_argument("--runs", type=int, default=10, help="runs por instancia standard")
    ap.add_argument("--large-runs", type=int, default=3, help="runs por instancia large")
    ap.add_argument("--no-large", action="store_true")
    ap.add_argument("--only", nargs="*", help="nombres de instancia a correr")
    ap.add_argument("--seed", type=int, default=1000, help="semilla base (run k usa seed+k)")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    bks = load_bks()
    items = collect(not args.no_large)
    if args.only:
        items = [it for it in items if it[1] in set(args.only)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    g._warmup()
    print(f"{len(items)} instancias | numba={g.NUMBA_OK} | stall={g.STALL_FACTOR}n | cap={g.TIME_CAP_S}s", flush=True)
    cols = ["instance", "n", "kind", "runs", "bks", "best", "avg", "gap_best", "gap_avg",
            "hits", "t_avg", "t_max", "kicks_avg"]
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    t_all = time.time()
    for n, name, path, kind in items:
        runs = args.large_runs if kind == "large" else args.runs
        nodes = g.read_tsplib(path)
        lens, times, kicks = [], [], []
        for r in range(runs):
            t0 = time.time()
            routes, _, L, _, _, info = g.gsph_fc(nodes, seed=args.seed + r)
            times.append(time.time() - t0)
            route = routes["QALL"]
            if len(route) != len(nodes) or sorted(route) != sorted(nodes):
                raise RuntimeError(f"{name}: tour inválido en run {r}")
            lens.append(L)
            kicks.append(info["kicks"])
        b = bks.get(name)
        best, avg = min(lens), sum(lens) / runs
        gb = 100.0 * (best - b) / b if b else float("nan")
        ga = 100.0 * (avg - b) / b if b else float("nan")
        hits = sum(1 for x in lens if b and x <= b)
        row = [name, n, kind, runs, b or "", best, f"{avg:.1f}", f"{gb:.3f}", f"{ga:.3f}", hits,
               f"{sum(times) / runs:.3f}", f"{max(times):.3f}", sum(kicks) // runs]
        with open(args.out, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"{name:10s} n={n:6d} best={best} BKS={b} gap_best={gb:.3f}% gap_avg={ga:.3f}% "
              f"hits={hits}/{runs} t_avg={sum(times) / runs:.2f}s", flush=True)
    print(f"Total {time.time() - t_all:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
