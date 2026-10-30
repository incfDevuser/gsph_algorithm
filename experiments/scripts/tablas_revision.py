"""
Genera las tablas LaTeX de la revisión del paper a partir de los CSV de los
experimentos (ablación, sensibilidad, LKH, implementación anterior re-medida).

Uso:  python experiments/scripts/tablas_revision.py
Salida: experiments/results/tablas_revision.tex (y un resumen por pantalla)
"""
import csv, os, sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(SCRIPT_DIR, "..", "results")
sys.path.insert(0, SCRIPT_DIR)
from benchmark_v2 import load_bks  # noqa: E402

RANGOS = [(1, 300, r"$n\le 300$"), (301, 800, r"$301\le n\le 800$"), (801, 2000, r"$801\le n\le 2{,}000$")]
NOMBRES = {"aleatorio": "Random + ILS", "vecino": "Nearest Neighbor + ILS",
           "particion": "Partition (no FC) + ILS", "fc": "Frontier Connectivity + ILS",
           "gsph_fc": "GSPH-FC (complete)"}


def leer(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def v2():
    return {r["instance"]: r for r in leer(os.path.join(RES, "v2", "benchmark_v2.csv"))}


def tabla_ablacion(out):
    p = os.path.join(RES, "ablacion", "ablacion_runs.csv")
    if not os.path.exists(p):
        return
    bks = load_bks()
    runs = leer(p)
    por = defaultdict(list)
    for r in runs:
        por[(r["config"], r["instance"])].append(r)
    insts = sorted({(int(r["n"]), r["instance"]) for r in runs})
    gap = lambda L, i: 100.0 * (L - bks[i]) / bks[i]
    filas = []
    for cfg in NOMBRES:
        raw, light, best, avg, t, hits, nruns = [], [], [], [], [], 0, 0
        por_rango = {k: [] for k in range(len(RANGOS))}
        for n, i in insts:
            rs = por.get((cfg, i))
            if not rs:
                continue
            L = [int(r["L_final"]) for r in rs]
            raw.append(np.mean([gap(int(r["L_raw"]), i) for r in rs]))
            light.append(np.mean([gap(int(r["L_light"]), i) for r in rs]))
            best.append(gap(min(L), i))
            avg.append(np.mean([gap(x, i) for x in L]))
            t.append(np.mean([float(r["t_total"]) for r in rs]))
            hits += sum(x <= bks[i] for x in L)
            nruns += len(L)
            for k, (lo, hi, _) in enumerate(RANGOS):
                if lo <= n <= hi:
                    por_rango[k].append((np.mean([gap(x, i) for x in L]), t[-1]))
        if not raw:
            continue
        filas.append((cfg, np.mean(raw), np.mean(light), np.mean(best), np.mean(avg), hits, nruns,
                      np.mean(t), por_rango))
    n_inst = len(insts)
    nr = filas[0][6] // n_inst if filas else 0
    out.append("% --- Ablación de la construcción (ablacion_construccion.py) ---")
    out.append(r"\begin{table}")
    out.append(r"\caption{Construction ablation: every variant is followed by the same "
               r"local search and ILS (same stopping rule, temperature, and seeds). "
               rf"{n_inst} instances with $n\le 2{{,}}000$, {nr} runs each. "
               r"GAP values are averages over instances.}")
    out.append(r"\label{tab:ablation-constr}")
    out.append(r"\centering\scriptsize")
    out.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    out.append(r"\toprule")
    out.append(r"\textbf{Construction} & \textbf{GAP seed} & \textbf{GAP LS} & \textbf{GAP best} & "
               r"\textbf{GAP avg} & \textbf{Hits} & \textbf{Time (s)}\\")
    out.append(r"\midrule")
    for cfg, raw, light, best, avg, hits, nruns, t, _ in filas:
        out.append(f"{NOMBRES[cfg]} & {raw:.2f} & {light:.2f} & {best:.3f} & {avg:.3f} & "
                   f"{hits}/{nruns} & {t:.2f} \\\\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{7}{@{}l}{\textit{GAP avg (\%) / time (s) by size range}}\\")
    for cfg, *_, por_rango in filas:
        celdas = []
        for k in range(len(RANGOS)):
            v = por_rango[k]
            celdas.append(f"{np.mean([a for a, _ in v]):.3f} / {np.mean([b for _, b in v]):.2f}" if v else "--")
        out.append(f"{NOMBRES[cfg]} & \\multicolumn{{2}}{{r}}{{{celdas[0]}}} & "
                   f"\\multicolumn{{2}}{{r}}{{{celdas[1]}}} & \\multicolumn{{2}}{{r}}{{{celdas[2]}}} \\\\")
    out.append(r"\multicolumn{1}{l}{} & \multicolumn{2}{r}{\scriptsize " + RANGOS[0][2] +
               r"} & \multicolumn{2}{r}{\scriptsize " + RANGOS[1][2] +
               r"} & \multicolumn{2}{r}{\scriptsize " + RANGOS[2][2] + r"}\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")
    for cfg, raw, light, best, avg, hits, nruns, t, pr in filas:
        print(f"ABL {cfg:10s} seed={raw:7.2f} LS={light:6.2f} best={best:.3f} avg={avg:.3f} "
              f"hits={hits}/{nruns} t={t:.2f} | " +
              " ".join(f"{np.mean([a for a, _ in pr[k]]):.3f}/{np.mean([b for _, b in pr[k]]):.2f}"
                       for k in range(len(RANGOS)) if pr[k]))


def tabla_sensibilidad(out):
    p = os.path.join(RES, "sensibilidad", "sens_resumen.csv")
    if not os.path.exists(p):
        return
    rows = leer(p)
    etiquetas = {"default": r"Default ($k=3$, portfolio, $T=0.2\bar d$, $n/8$, $300n$)",
                 "FRONTIER_TOPK": r"Frontier pairs $k$", "PARTITION_PORTFOLIO_LIMIT": "Partitions in portfolio",
                 "ACCEPT_TEMP": r"Temperature $T/\bar d$", "DB_WINDOW_DIV": r"Window $W(n)=\max(50,n/\cdot)$",
                 "STALL_FACTOR": r"Stall limit $S(n)=\max(5000,\cdot\,n)$"}
    out.append("% --- Sensibilidad de parámetros (sensibilidad_parametros.py) ---")
    out.append(r"\begin{table}")
    out.append(r"\caption{One-at-a-time parameter sensitivity on the tuning subset "
               r"(\texttt{kroE100}, \texttt{lin318}, \texttt{pcb442}, \texttt{rat783}, \texttt{pr1002}, "
               r"\texttt{d1655}); GAP values averaged over the six instances.}")
    out.append(r"\label{tab:sensitivity}")
    out.append(r"\centering\scriptsize")
    out.append(r"\begin{tabular}{@{}llrrrr@{}}")
    out.append(r"\toprule")
    out.append(r"\textbf{Parameter} & \textbf{Value} & \textbf{GAP best} & \textbf{GAP avg} & "
               r"\textbf{Hits} & \textbf{Time (s)}\\")
    out.append(r"\midrule")
    for r in rows:
        if r["variant"] == "default":
            par, val = etiquetas["default"], "--"
        else:
            p_, v_ = r["variant"].split("=")
            par, val = etiquetas[p_], v_
            if p_ == "PARTITION_PORTFOLIO_LIMIT":
                val = "1 (bbox only)"
        out.append(f"{par} & {val} & {float(r['gap_best_avg']):.3f} & {float(r['gap_avg_avg']):.3f} & "
                   f"{r['hits']} & {float(r['t_avg']):.2f} \\\\")
        print("SENS", r["variant"], r["gap_best_avg"], r["gap_avg_avg"], r["hits"], r["t_avg"])
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def tabla_lkh(out):
    p = os.path.join(RES, "lkh", "lkh_runs.csv")
    if not os.path.exists(p):
        return
    bks, g2 = load_bks(), v2()
    runs = leer(p)
    por = defaultdict(list)
    for r in runs:
        por[(r["mode"], r["instance"])].append(r)
    insts = sorted({(int(r["n"]), r["instance"]) for r in runs})
    nr = max(len(v) for v in por.values())
    out.append("% --- Comparación con LKH (comparar_lkh.py) ---")
    out.append(r"\begin{table}")
    out.append(r"\caption{GSPH-FC versus LKH-3 by size range. GSPH-FC: 10 runs (Tables~\ref{tab:small}--"
               rf"\ref{{tab:large}}). LKH: {nr} runs with default parameters (one run of $n$ trials per "
               r"seed) and with a time limit equal to the average GSPH-FC time on each instance. "
               r"Hits count runs that reach the BKS.}")
    out.append(r"\label{tab:lkh}")
    out.append(r"\centering\scriptsize")
    out.append(r"\begin{tabular}{@{}lrrrrrrrrr@{}}")
    out.append(r"\toprule")
    out.append(r"& \multicolumn{3}{c}{\textbf{GSPH-FC}} & \multicolumn{3}{c}{\textbf{LKH default}} & "
               r"\multicolumn{3}{c}{\textbf{LKH, same time}}\\")
    out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    out.append(r"\textbf{Range} & \textbf{GAP avg} & \textbf{Hits} & \textbf{Time} & \textbf{GAP avg} & "
               r"\textbf{Hits} & \textbf{Time} & \textbf{GAP avg} & \textbf{Hits} & \textbf{Time}\\")
    out.append(r"\midrule")
    for lo, hi, etq in RANGOS + [(1, 2000, r"\textit{All}")]:
        sel = [i for n, i in insts if lo <= n <= hi]
        if not sel:
            continue
        celdas = [f"{np.mean([float(g2[i]['gap_avg']) for i in sel]):.3f}",
                  f"{sum(int(g2[i]['hits']) for i in sel)}/{sum(int(g2[i]['runs']) for i in sel)}",
                  f"{np.mean([float(g2[i]['t_avg']) for i in sel]):.2f}"]
        for modo in ("default", "mismo_t"):
            ga, h, tt, nn = [], 0, [], 0
            for i in sel:
                rs = por.get((modo, i), [])
                if not rs:
                    continue
                L = [int(r["L"]) for r in rs]
                ga.append(np.mean([100.0 * (x - bks[i]) / bks[i] for x in L]))
                h += sum(x <= bks[i] for x in L)
                nn += len(L)
                tt.append(np.mean([float(r["t"]) for r in rs]))
            celdas += [f"{np.mean(ga):.3f}", f"{h}/{nn}", f"{np.mean(tt):.2f}"] if ga else ["--"] * 3
        if lo == 1 and hi == 2000:
            out.append(r"\midrule")
        out.append(f"{etq} & " + " & ".join(celdas) + r" \\")
        print("LKH", etq, celdas)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")


def tabla_v1(out):
    p = os.path.join(RES, "v1_remedido", "v1_37.csv")
    if not os.path.exists(p):
        return
    g2 = v2()
    rows = leer(p)
    out.append("% --- Implementación anterior re-medida (remedir_v1.py) ---")
    out.append(r"\begin{tabular}{@{}lrrrrrrr@{}}")
    tot = defaultdict(list)
    for r in sorted(rows, key=lambda r: int(g2[r["instance"]]["n"])):
        i = r["instance"]
        c = g2[i]
        sp = float(r["t_avg"]) / float(c["t_avg"])
        tot["g1"].append(float(r["gap_best"]))
        tot["t1"].append(float(r["t_avg"]))
        tot["gb"].append(float(c["gap_best"]))
        tot["ga"].append(float(c["gap_avg"]))
        tot["t2"].append(float(c["t_avg"]))
        tot["h1"].append(float(r["gap_best"]) == 0)
        tot["h2"].append(float(c["gap_best"]) == 0)
        if float(r["gap_best"]) > 0:
            gb = r"\textbf{0.000}" if float(c["gap_best"]) == 0 else f"{float(c['gap_best']):.3f}"
            out.append(f"{i} & {c['n']} & {float(r['gap_best']):.3f} & {float(r['t_avg']):.2f} & {gb} & "
                       f"{float(c['gap_avg']):.3f} & {float(c['t_avg']):.2f} & {sp:.0f}$\\times$ \\\\")
    sp_tot = np.mean(tot["t1"]) / np.mean(tot["t2"])
    out.append(r"\midrule")
    out.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{All {len(rows)} instances}}}} & {np.mean(tot['g1']):.3f} & "
               f"{np.mean(tot['t1']):.2f} & {np.mean(tot['gb']):.3f} & {np.mean(tot['ga']):.3f} & "
               f"{np.mean(tot['t2']):.2f} & {sp_tot:.0f}$\\times$ \\\\")
    out.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{BKS hits}}}} & \\multicolumn{{2}}{{c}}{{{sum(tot['h1'])} / "
               f"{len(rows)}}} & \\multicolumn{{3}}{{c}}{{{sum(tot['h2'])} / {len(rows)}}} & \\\\")
    out.append(r"\end{tabular}")
    out.append("")
    print(f"V1 {len(rows)} inst: gap_best={np.mean(tot['g1']):.3f} t={np.mean(tot['t1']):.2f} "
          f"hits={sum(tot['h1'])} | V2 t={np.mean(tot['t2']):.2f} speedup={sp_tot:.0f}x")


def main():
    out = []
    tabla_ablacion(out)
    tabla_sensibilidad(out)
    tabla_lkh(out)
    tabla_v1(out)
    path = os.path.join(RES, "tablas_revision.tex")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
