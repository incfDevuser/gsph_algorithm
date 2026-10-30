# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research repo for **GSPH** (Geometric Split and Polish Heuristic), a 2D Euclidean TSP heuristic.
Two halves: a Python experiment harness in `experiments/` and an LLNCS-class LaTeX paper at the root
(`samplepaper_v2.tex` is the current paper, about GSPH_V2, and `samplepaper_v2_es.tex` its Spanish
translation with the same tables — keep both in sync; `samplepaper.tex` is the previous version
and `test.tex` an older draft; all cite `bibliografia.bib`).
Code comments, output files and docs are in **Spanish**; keep new ones in Spanish. `AGENTS.md` holds
the same operating notes for other agents; keep the two in sync when either changes.

## Setup
`.python-version` says 3.11.5, but the existing `.venv` runs Python 3.14 (with numba 0.68.1).
Always use the repo venv — the system `python3` lacks matplotlib, so plots silently get skipped.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
`scipy`, `matplotlib`, `psutil`, `pympler`, `memory_profiler` are imported in `try/except` in
`GSPH_BASE.py`; the solver degrades (e.g. no Delaunay candidates, brute-force kNN, no plots) instead of
failing. `GSPH_V2.py` treats `numba` the same way. `compare_runs.py` additionally needs Pillow (present in `.venv`, not in `requirements.txt`).

## Commands
There is no build step, test suite, linter config or CI. "Testing" = running instances and checking
the GAP vs BKS.

`GSPH_BASE.py` uses relative paths (`../instances/standard`, `../results`), so **run it from
`experiments/algorithms/`**:
```bash
cd experiments/algorithms
python3 GSPH_BASE.py                      # every .tsp in ../instances/standard (74 instances)
python3 GSPH_BASE.py berlin52             # one instance; name, file.tsp, or unique substring
python3 GSPH_BASE.py --profile berlin52   # best-of-N runs, N = GSPH_PROFILE_RUNS (default 10)
GSPH_LIMIT=5 python3 GSPH_BASE.py         # first 5 instances (alphabetical)
GSPH_QUALITY_TIME_CAP=10 python3 GSPH_BASE.py a280   # cap strong-polish seconds per run (default 30)
```
Quick smoke check: `python3 GSPH_BASE.py berlin52` (BKS 7542, ~5 s strong-polish budget). A crash is
not printed as a traceback — the run prints `Failed …: runtime error` and the message goes to
`results/<inst>/error.txt`; call `gsph_fc` directly to get the stack.

Full paper benchmark (runs from any cwd; ~74 instances × 20 runs; hours with BASE, much less with V2):
```bash
python3 experiments/scripts/run_all_and_latex.py                  # writes experiments/results/tablas_gsph_base.tex
python3 experiments/scripts/run_all_and_latex.py --algo GSPH_V2   # writes experiments/results/tablas_gsph_v2.tex
python3 experiments/scripts/compare_runs.py experiments/results/berlin52 --cols 5
```
For iterating on the algorithm, prefer the fast in-process runner (no plots, validates every tour,
GAP vs `BKS_TSP.txt`, CSV to `experiments/results/v2/benchmark_v2.csv`). It covers `standard/` plus the
EUC_2D files in `large/` (the `pla*` ones are CEIL_2D and are skipped):
```bash
python3 experiments/scripts/benchmark_v2.py --runs 10 --large-runs 3
python3 experiments/scripts/benchmark_v2.py --runs 5 --no-large --only kroE100 lin318 pcb442 rat783 pr1002 d1655
```
That 6-instance subset is a good quick tuning set (it exposed the stuck-basin and window issues).
Only `instances/standard/` is used by the solver; `instances/large/` (2k–86k nodes) is not wired in
and would not fit the dense distance matrix anyway.

## Solver architecture (`experiments/algorithms/GSPH_BASE.py`)
`gsph_fc(nodes, MAX_ITER_LOCAL, EPS_FRONTIER)` is the whole pipeline; it returns
`(routes, connections, total_len, xmid, ymid)` where the final tour is `routes['QALL']`.

1. **Precompute once**: dense integer distance matrix `D` (`build_distance_matrix`, O(n²) memory) and
   candidate neighbour lists `cand` (kNN via `cKDTree` ∪ Delaunay edges, symmetrised, truncated to
   `m`). `polish_cand_knn` bumps kNN to 40 for n > 250.
2. **Split — partition portfolio**: `generate_partition_portfolio` produces several (xmid, ymid) cut
   points (bbox centre, median, four shifted centres). Each splits points into quadrants Q1..Q4,
   visited in the fixed cycle `Q_ORDER = Q1→Q2→Q4→Q3` (`ADJACENCY_CYCLE`).
3. **Assembly candidates**: for each adjacent quadrant pair, `best_frontier_pairs` picks the top-k
   node pairs across the frontier; `_enumerate_frontier_choice_maps` combines them; each combination
   gets per-quadrant anchored paths (`build_anchored_path`, local 2-opt with fixed endpoints) that are
   concatenated into a raw tour.
4. **Funnel**: raw tours are prefiltered per partition, pooled, given a cheap `polish_light`
   (2-opt DLB + Or-opt, no ILS), then 1–2 finalists (second only if within `FINALIST_MARGIN_RATIO`)
   get the strong polish. How wide each stage is depends on n via `get_search_profile`.
5. **Polish — `polish_fast` → `_polish_quality`**: time-driven ILS. Initial local optimum with
   queue-based 2-opt + don't-look bits (`_two_opt_queue`) and Or-opt with reversed insertion; then
   loop until deadline: `double_bridge_tracked` kick, re-optimise only the perturbed nodes + their
   candidates, accept if ≤ current, Or-opt intensification on new bests, restart from best after 30
   fails. Deadline is `quality_time_budget(n)` (5/12/25/30 s by size, capped by
   `GSPH_QUALITY_TIME_CAP`).

Representation detail: routes/tours are lists of coordinate tuples; local search works on index
permutations. `route_to_perm` maps by coordinate value, so duplicate coordinates collapse;
`ensure_valid_perm` repairs any missing/duplicate indices by cheapest insertion. Keep that round-trip
in mind when changing either representation.

### Tunables that actually matter
Module constants at the top of the file. Several are **stale/unused** in the current single-mode
pipeline: `ILS_ITERS` and `MAX_SWEEPS_2OPT` are not referenced, and `TIME_BUDGET_S` only scales the
light-polish budget. The strong polish is governed by `quality_time_budget`, `OR_OPT_MAX_L`,
`CAND_M`/`polish_cand_knn`, and the funnel by `get_search_profile`. The parameter table in
`experiments/README.md` is outdated.

`random` is never seeded, so runs are not reproducible; `--profile` best-of-N relies on that variance.

## GSPH_V2 (`experiments/algorithms/GSPH_V2.py`)
Faster, stronger rewrite of the same pipeline; `GSPH_BASE.py` is kept as the "previous
implementation" of the paper and only received a crash fix (`double_bridge_tracked` indexed `perm[n]`
when n is divisible by 4; runs that didn't crash are unchanged). Same CLI and stdout format as BASE (so `run_all_and_latex.py`
parses it), but results go to `experiments/results/v2/`.
- Works on node indices end to end (no coordinate-tuple round-trip), so duplicate coordinates are safe.
- Kernels are `@njit(cache=True)` numba functions over numpy arrays: `_local_search` (queue-driven
  2-opt in both directions with neighbour-list pruning + Or-opt L≤3, both orientations; returns the
  total gain so lengths are tracked incrementally), `_ils_block` (local double-bridge in a window of
  `max(DB_WINDOW_MIN, n/DB_WINDOW_DIV)` positions; accept if ≤ current, or worse with probability
  exp(−Δ/T), T = `ACCEPT_TEMP`·avg edge, env `GSPH_TEMP`, default 0.2; keeps the best tour seen),
  `_anchored_path` (cheapest insertion with lazy re-evaluation of stale costs + fixed-endpoint 2-opt).
- Every tour change in the local search is a reversal (`_rev_logged`; Or-opt = 2–3 reversals via
  `_or_move`), logged in a journal so a rejected kick is undone in reverse order instead of copying
  the tour. If the journal (`JOURNAL_SIZE`) overflows, it falls back to the best-tour copy. Any new
  move type must go through `_rev_logged`/`_two_opt_move` or rejections will corrupt the tour.
- `build_distance_matrix` is computed in row blocks (peak ≈ n²·4 bytes), so instances up to ~18k
  nodes fit; beyond that the dense matrix itself is the limit. For n ≳ 10k the array-based 2-opt
  (O(n) reversals) dominates and runs hit `GSPH_TIME_CAP`.
- Without numba the decorator is a no-op and the same code runs as plain Python (slow but correct);
  `NUMBA_DISABLE_JIT=1` forces that path for testing.
- Numba can't read the clock, so `polish_ils` calls `_ils_block` in chunks of `ILS_BLOCK` kicks and
  stops on **stagnation**: `max(STALL_MIN, STALL_FACTOR·n)` kicks without strict improvement
  (`GSPH_STALL` overrides), with `GSPH_TIME_CAP` (default 60 s) as a safety cap. It never uses the BKS.
- `GSPH_SEED` makes runs reproducible (run k uses seed+k); the seed and kick count go to `resumen.txt`.
- `_warmup()` compiles/loads the kernels before timing; the first run after editing a kernel pays
  a few seconds of compilation.
- `gsph_fc` returns a 6th element (`info` dict) and raises if the tracked length disagrees with the
  recomputed `total_tour_length`, so a broken move shows up immediately.

## Metric and BKS
Reported lengths use TSPLIB **EUC_2D rounded to int** (`int(d + 0.5)`) via `euclidean_distance`, `D`,
and `total_tour_length`. `eucl_float` is only for internal heuristic decisions — never report it.
The parser reads only `NODE_COORD_SECTION` (EUC_2D assumed; ATT/GEO instances would be scored wrong).
BKS live in two places: the `BKS` dict in `run_all_and_latex.py` (used for GAPs) and
`experiments/benchmarks/BKS/BKS_TSP.txt`.

## Outputs and cross-file coupling
- Per instance: `experiments/results/<inst>/resumen.txt`, `estadisticas_runs.txt` (profile),
  `graficos/{ruta_final,ensamblaje_cuadrantes,orden_visita}.png`, and under `--profile`
  `graficos/comparacion_runs.png` + `graficos/runs/run_XX[_best].png`. `_prepare_output_dirs` wipes
  previous files in that instance dir on every run.
- `experiments/results/summary.csv` is **rewritten** (not appended) by each `GSPH_BASE.py` invocation.
- `results/*` is git-ignored except `.gitkeep`.
- `run_all_and_latex.py` parses `GSPH_BASE.py` **stdout** with regexes — the per-run
  `length=…, time=…s` lines and the final `BEST=… (avg=…)` line. Changing those print formats breaks
  the benchmark tables.
- `compare_runs.py` globs `graficos/runs/` for `run_*.png` / `*_run_*.png`; keep plot names aligned.

## Paper revision experiments (`experiments/scripts/`)
Each table added in the review of `samplepaper_v2.tex` has its own script (all take `--workers`, write
long-format CSV + summary under `experiments/results/<dir>/`, and validate every tour):
- `ablacion_construccion.py` → `ablacion/`: five constructions (random, NN, partition without
  frontiers, FC with k=1, full GSPH-FC) followed by the same V2 polish; `gsph_fc_completo` replicates
  `GSPH_V2.gsph_fc` exactly (same seed → same length) while logging seed/LS/final lengths.
- `sensibilidad_parametros.py` → `sensibilidad/`: one-at-a-time changes to V2 module constants on the
  6-instance tuning set (sets the globals inside each worker).
- `comparar_lkh.py` → `lkh/`: LKH default vs same time as GSPH-FC. Uses an official LKH binary in
  `experiments/data/lkh/` if present, else LKH-3.0.8 embedded in `elkai` via `_elkai.solve_problem`
  (passes SEED/TIME_LIMIT). On Python 3.14 elkai only builds with
  `PIP_CONSTRAINT` containing `scikit-build-core<0.10`.
- `remedir_v1.py` → `v1_remedido/`: re-runs GSPH_BASE best-of-5; a partial run (30/37) confirmed the
  paper's previous-implementation numbers (`v1_verificacion_parcial.csv`).
- `tablas_revision.py`: turns those CSVs into LaTeX (`experiments/results/tablas_revision.tex`).
Run experiments one at a time when timings matter; times taken with other jobs running are only
comparable within the same table. `eil10`/`eil20` are subsets of eil51 (optima 159/243 by Held-Karp).

## Other files
- `GSPH_V1.py`: old baseline kept for comparison; don't develop on it.
- `verify_structure.py` lists some scripts that don't exist — aspirational, not authoritative.
- `experiments/README.md` references `run_all_instances.py` and `aggregate_resources.py`, which do
  not exist, and an older output layout (`resultados.txt`, `grafico_*.png`).
- `samplepaper_v2.tex` takes its figures from `Resultados_v2/` (in the repo); the old
  `samplepaper.tex` points to `Resultados/*.png`, which is not. No LaTeX toolchain is installed locally.
