"""
Vuelve a medir la implementación anterior (GSPH_BASE) en la misma máquina y
software que GSPH_V2, sobre las 37 instancias de la comparación del paper
(Tabla "Previous vs. current"), con el mismo protocolo: mejor de 5 runs.

Uso (desde cualquier carpeta):
  python experiments/scripts/remedir_v1.py              # las 37 instancias
  python experiments/scripts/remedir_v1.py --only rat99 pcb442

Escribe experiments/results/v1_remedido/v1_37.csv. GSPH_BASE borra y reescribe
experiments/results/<inst>/ en cada corrida.
"""
import argparse, csv, os, platform, re, subprocess, sys, time
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(SCRIPT_DIR, "..", "algorithms")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "v1_remedido")
sys.path.insert(0, SCRIPT_DIR)
from run_all_and_latex import BKS  # noqa: E402

INSTANCIAS_37 = [
    "eil51", "berlin52", "st70", "eil76", "pr76", "rat99", "kroA100", "kroB100", "kroC100",
    "kroD100", "kroE100", "rd100", "eil101", "lin105", "pr107", "pr124", "bier127", "ch130",
    "pr136", "pr144", "ch150", "kroA150", "kroB150", "pr152", "u159", "d198", "kroA200",
    "kroB200", "ts225", "tsp225", "pr226", "gil262", "pr264", "a280", "pr299", "lin318", "pcb442",
]
RUNS = 5


def correr(nombre):
    env = os.environ.copy()
    env["GSPH_PROFILE_RUNS"] = str(RUNS)
    cmd = [sys.executable, os.path.join(ALGO_DIR, "GSPH_BASE.py"), "--profile", nombre]
    t0 = time.monotonic()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ALGO_DIR, env=env, timeout=7200)
    texto = out.stdout + out.stderr
    # Líneas por run: "Run k/N... length=426.00, time=2.056s" (la final trae best_time=, que no cuenta).
    pares = re.findall(r"length=([\d.]+), time=([\d.]+)s", texto)
    largos = [float(l) for l, _ in pares]
    tiempos = [float(t) for _, t in pares]
    if len(largos) != RUNS or len(tiempos) != RUNS:
        raise RuntimeError(f"{nombre}: se esperaban {RUNS} runs, se leyeron {len(largos)}/{len(tiempos)}")
    return largos, tiempos, time.monotonic() - t0


def fila_instancia(nombre):
    largos, tiempos, _ = correr(nombre)
    bks = BKS[nombre]
    best, avg = min(largos), sum(largos) / len(largos)
    return [nombre, bks, RUNS, int(best), f"{avg:.1f}", f"{(best - bks) / bks * 100:.3f}",
            f"{(avg - bks) / bks * 100:.3f}", sum(int(l) == bks for l in largos),
            f"{sum(tiempos) / len(tiempos):.2f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=1,
                    help="instancias en paralelo (GSPH_BASE para por presupuesto de tiempo, "
                         "así que el tiempo por run casi no cambia)")
    ap.add_argument("--reanudar", action="store_true", help="salta las instancias ya escritas en el CSV")
    args = ap.parse_args()
    nombres = args.only or INSTANCIAS_37
    os.makedirs(OUT_DIR, exist_ok=True)
    ruta = os.path.join(OUT_DIR, "v1_37.csv" if not args.only else "v1_parcial.csv")
    hechas = set()
    if args.reanudar and os.path.exists(ruta):
        with open(ruta) as f:
            hechas = {r["instance"] for r in csv.DictReader(f)}
    else:
        with open(ruta, "w", newline="") as f:
            csv.writer(f).writerow(["instance", "bks", "runs", "best", "avg", "gap_best", "gap_avg",
                                    "hits", "t_avg"])
    pendientes = [x for x in nombres if x not in hechas]
    with Pool(args.workers) as pool:
        for fila in pool.imap_unordered(fila_instancia, pendientes):
            with open(ruta, "a", newline="") as f:
                csv.writer(f).writerow(fila)
            print(" ".join(map(str, fila)), flush=True)
    with open(os.path.join(OUT_DIR, "entorno.txt"), "w") as f:
        f.write(f"{platform.platform()}\n{platform.processor()}\nPython {platform.python_version()}\n")
    print(f"Escrito {ruta}")


if __name__ == "__main__":
    main()
