"""
Ejecuta TODAS las instancias del paper con GSPH_BASE
y genera tablas LaTeX con los GAPs vs BKS.
"""

import subprocess, sys, os, time, re

SMALL = [
    "eil10", "eil20", "eil51", "eil76", "eil101",
    "berlin52", "st70", "pr76", "rat99", "rd100",
    "kroA100", "kroB100", "kroC100", "kroD100", "kroE100",
    "lin105", "pr107",
]

MEDIUM = [
    "pr124", "bier127", "ch130", "pr136", "ch150",
    "pr144", "kroA150", "kroB150", "pr152", "u159",
    "rat195", "d198", "kroA200", "kroB200", "ts225",
    "tsp225", "pr226", "pr264", "a280", "gil262", "pr299",
]

INTERMEDIATE = [
    "lin318", "linhp318", "rd400", "fl417", "pr439",
    "pcb442", "d493", "d657", "u574", "rat575",
    "p654", "u724", "rat783",
]

LARGE = [
    "pr1002", "u1060", "vm1084", "pcb1173", "d1291",
    "rl1304", "rl1323", "nrw1379", "fl1400", "u1432",
    "fl1577", "d1655", "u1817", "vm1748", "rl1889",
]

VERY_LARGE = [
    "u2152", "u2319", "pr2392", "pcb3038", "fl3795",
    "fnl4461", "rl5915", "rl5934",
]

ALL_INSTANCES = SMALL + MEDIUM + INTERMEDIATE + LARGE + VERY_LARGE

BKS = {
    "a280":2579, "berlin52":7542, "bier127":118282, "ch130":6110,
    "ch150":6528, "d198":15780, "d493":35002, "d657":48912,
    "d1291":50801, "d1655":62128, "eil51":426, "eil76":538,
    "eil101":629, "fl417":11861, "fl1400":20127, "fl1577":22249,
    "fl3795":28772, "fnl4461":182566, "gil262":2378, "kroA100":21282,
    "kroB100":22141, "kroC100":20749, "kroD100":21294, "kroE100":22068,
    "kroA150":26524, "kroB150":26130, "kroA200":29368, "kroB200":29437,
    "lin105":14379, "lin318":42029, "linhp318":41345, "nrw1379":56638,
    "p654":34643, "pcb442":50778, "pcb1173":56892, "pcb3038":137694,
    "pr76":108159, "pr107":44303, "pr124":59030, "pr136":96772,
    "pr144":58537, "pr152":73682, "pr226":80369, "pr264":49135,
    "pr299":48191, "pr439":107217, "pr1002":259045, "pr2392":378032,
    "rat99":1211, "rat195":2323, "rat575":6773, "rat783":8806,
    "rd100":7910, "rd400":15281, "rl1304":252948, "rl1323":270199,
    "rl1889":316536, "rl5915":565530, "rl5934":556045, "st70":675,
    "ts225":126643, "tsp225":3916, "u159":42080, "u574":36905,
    "u724":41910, "u1060":224094, "u1432":152970, "u1817":57201,
    "u2152":64253, "u2319":234256, "vm1084":239297, "vm1748":336556,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(SCRIPT_DIR, "..", "algorithms")
GSPH_BASE = os.path.join(ALGO_DIR, "GSPH_BASE.py")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
LATEX_OUT = os.path.join(RESULTS_DIR, "tablas_gsph_base.tex")
RUNS = 20


def run_instance(name):
    cmd = [sys.executable, GSPH_BASE, "--profile", name]
    print(f"\n{'='*60}")
    print(f"  Ejecutando: {name} ({RUNS} runs)")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, cwd=ALGO_DIR,
        )
        output = result.stdout + result.stderr
        lines_out = output.strip().split('\n')
        for line in lines_out[-5:]:
            print(f"  {line}")

        m = re.search(
            r"BEST=([\d.]+)\s*\(avg=([\d.]+)\).*?"
            r"gap_best=([\d.-]+)%.*?"
            r"gap_avg=([\d.-]+)%.*?"
            r"best_time=([\d.]+)s",
            output, re.DOTALL
        )
        if m:
            return {
                "best": float(m.group(1)),
                "avg": float(m.group(2)),
                "gap_best": float(m.group(3)),
                "gap_avg": float(m.group(4)),
                "time": float(m.group(5)),
            }

        lengths = [float(x) for x in re.findall(r"length=([\d.]+)", output)]
        times_list = [float(x) for x in re.findall(r"time=([\d.]+)s", output)]
        if lengths:
            best = min(lengths)
            avg = sum(lengths) / len(lengths)
            t = sum(times_list) / len(times_list) if times_list else 0
            bks = BKS.get(name)
            gb = ((best - bks) / bks * 100) if bks else 0
            ga = ((avg - bks) / bks * 100) if bks else 0
            return {"best": best, "avg": avg, "gap_best": gb, "gap_avg": ga, "time": t}

        print(f"  ** No se pudo parsear resultado para {name}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ** Timeout para {name}")
        return None
    except Exception as e:
        print(f"  ** Error para {name}: {e}")
        return None


def get_nodes(name):
    for subdir in ["standard", "large"]:
        path = os.path.join(SCRIPT_DIR, "..", "instances", subdir, f"{name}.tsp")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip().upper().startswith("DIMENSION"):
                        m = re.search(r"\d+", line)
                        if m:
                            return int(m.group())
    return 0


def make_latex_table(caption, instances, results, footnote=None):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{l S[table-format=5.0] S[table-format=7.0] S[table-format=7.0] S[table-format=-2.3] S[table-format=4.3]}")
    lines.append(r"\toprule")
    lines.append(r"{Instancia} & {$n$} & {GSPH} & {BKS} & {GAP (\%)} & {Tiempo (s)} \\")
    lines.append(r"\midrule")

    gaps_valid = []
    sorted_inst = []
    for name in instances:
        r = results.get(name)
        if r is None:
            continue
        n = get_nodes(name)
        bks = BKS.get(name)
        gap = r["gap_best"]
        sorted_inst.append((gap, name, n, r, bks))
    sorted_inst.sort(key=lambda x: x[0])

    for gap, name, n, r, bks in sorted_inst:
        best_int = int(round(r["best"]))
        t = r["time"]
        if bks is None:
            bks_str = r"\multicolumn{1}{c}{--}"
            gap_s = r"\multicolumn{1}{c}{--}"
        else:
            bks_str = str(bks)
            gap_s = f"{gap:.3f}"
            if name != "d198" and gap > -50:
                gaps_valid.append(gap)

        if bks and abs(gap) <= 0.05:
            lines.append(
                f"\\textbf{{{name}}} & \\textbf{{{n}}} & "
                f"\\textbf{{{best_int}}} & \\textbf{{{bks_str}}} & "
                f"\\textbf{{{gap_s}}} & \\textbf{{{t:.3f}}} \\\\"
            )
        elif name == "d198":
            lines.append(
                f"{name}$^{{\\dagger}}$ & {n} & {best_int} & {bks_str} & "
                f"{{{gap_s}}} & {t:.3f} \\\\"
            )
        else:
            lines.append(
                f"{name} & {n} & {best_int} & {bks_str} & "
                f"{gap_s} & {t:.3f} \\\\"
            )

    if gaps_valid:
        avg_gap = sum(gaps_valid) / len(gaps_valid)
        lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{4}}{{l}}{{\\textit{{Promedio GAP:}}}} & "
            f"{avg_gap:.3f} & \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if footnote:
        lines.append(r"\vspace{2pt}")
        lines.append(f"\\raggedright\\footnotesize {footnote}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_summary_table(all_results):
    groups = [
        ("Pequeñas ($n \\le 120$)", SMALL),
        ("Medianas ($121 \\le n \\le 300$)", MEDIUM),
        ("Intermedias ($301 \\le n \\le 800$)", INTERMEDIATE),
        ("Grandes ($801 \\le n \\le 2000$)", LARGE),
        ("Muy grandes ($n > 2000$)", VERY_LARGE),
    ]
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Resumen de rendimiento GSPH por rango de tamaño.}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l S[table-format=2.0] S[table-format=2.3] S[table-format=2.3] S[table-format=2.3] S[table-format=4.1]}")
    lines.append(r"\toprule")
    lines.append(r"{Rango} & {$k$} & {GAP prom.(\%)} & {GAP m\'in.(\%)} & {GAP m\'ax.(\%)} & {T prom.(s)} \\")
    lines.append(r"\midrule")

    all_gaps = []
    all_times = []
    for label, inst_list in groups:
        gaps = []
        tms = []
        for name in inst_list:
            r = all_results.get(name)
            if r and BKS.get(name) and name != "d198":
                gaps.append(r["gap_best"])
                tms.append(r["time"])
                all_gaps.append(r["gap_best"])
                all_times.append(r["time"])
        if gaps:
            lines.append(
                f"{label} & {len(gaps)} & {sum(gaps)/len(gaps):.3f} & "
                f"{min(gaps):.3f} & {max(gaps):.3f} & {sum(tms)/len(tms):.1f} \\\\"
            )

    if all_gaps:
        lines.append(r"\midrule")
        lines.append(
            f"\\textbf{{Global}} & \\textbf{{{len(all_gaps)}}} & "
            f"\\textbf{{{sum(all_gaps)/len(all_gaps):.3f}}} & "
            f"\\textbf{{{min(all_gaps):.3f}}} & "
            f"\\textbf{{{max(all_gaps):.3f}}} & "
            f"\\textbf{{{sum(all_times)/len(all_times):.1f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("  GSPH_BASE - Benchmark completo TSPLIB")
    print(f"  {len(ALL_INSTANCES)} instancias x {RUNS} runs")
    print("=" * 60)

    all_results = {}
    t0 = time.time()
    progress_path = os.path.join(RESULTS_DIR, "progress_base.txt")
    with open(progress_path, "w") as f:
        f.write(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")

    for i, name in enumerate(ALL_INSTANCES):
        print(f"\n[{i+1}/{len(ALL_INSTANCES)}]", end="")
        r = run_instance(name)
        if r:
            all_results[name] = r
            bks = BKS.get(name, 0)
            with open(progress_path, "a") as f:
                f.write(f"{name:12s}: BEST={r['best']:>10.0f}  BKS={bks:>10d}  GAP={r['gap_best']:>7.3f}%  t={r['time']:.1f}s\n")

    elapsed = time.time() - t0
    print(f"\n\nTiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Completadas: {len(all_results)}/{len(ALL_INSTANCES)}")

    latex_parts = []
    latex_parts.append(f"% GSPH_BASE - {RUNS} runs - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    latex_parts.append("% Requiere: booktabs, siunitx, float")
    latex_parts.append("")
    latex_parts.append(make_latex_table(
        "Instancias pequeñas ($n \\le 120$). En \\textbf{negrita} GAP $\\le 0.05\\%$.",
        SMALL, all_results))
    latex_parts.append(make_latex_table(
        "Instancias medianas ($121 \\le n \\le 300$). En \\textbf{negrita} GAP $\\le 0.05\\%$.",
        MEDIUM, all_results,
        footnote=r"$^{\dagger}$ GAP negativo por desajuste de métrica (\texttt{CEIL\_2D} vs.\ \texttt{EUC\_2D})."))
    latex_parts.append(make_latex_table(
        "Instancias intermedias ($301 \\le n \\le 800$).",
        INTERMEDIATE, all_results))
    latex_parts.append(make_latex_table(
        "Instancias grandes ($801 \\le n \\le 2000$).",
        LARGE, all_results))
    latex_parts.append(make_latex_table(
        "Instancias muy grandes ($n > 2000$).",
        VERY_LARGE, all_results))
    latex_parts.append(make_summary_table(all_results))

    full_latex = "\n".join(latex_parts)
    with open(LATEX_OUT, "w") as f:
        f.write(full_latex)

    print(f"\n Tablas LaTeX: {LATEX_OUT}")
    print(f"\n{'='*70}")
    print(f"{'Instancia':>12s}  {'BEST':>10s}  {'BKS':>10s}  {'GAP%':>8s}  {'Tiempo':>8s}")
    print(f"{'='*70}")
    for name in ALL_INSTANCES:
        r = all_results.get(name)
        if r:
            bks = BKS.get(name, 0)
            print(f"  {name:12s}  {r['best']:>10.0f}  {bks:>10d}  {r['gap_best']:>7.3f}%  {r['time']:>7.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
