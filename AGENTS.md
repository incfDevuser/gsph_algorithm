# AGENTS.md

Research repo for the **GSPH** (Geometric Split and Polish Heuristic) TSP algorithm.
Two parts: Python experiment harness under `experiments/`, and a LaTeX paper at the root
(`samplepaper.tex`, `test.tex`, `bibliografia.bib`). Code comments and docs are in Spanish.

## Setup
- Python **3.11.5** (`.python-version`). Use the repo venv:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt   # numpy, scipy, matplotlib, psutil, memory-profiler, pympler, pylint, numba
  ```
- Heavy deps (`scipy`, `matplotlib`, `psutil`, `pympler`, `memory_profiler`) are imported in
  `try/except`; the algorithm degrades gracefully if a dep is missing. Don't assume they're loaded.

## Running the algorithm — working directory matters
`GSPH_BASE.py` uses **relative paths** (`BASE_DIR="../instances/standard"`, `RESULTS_DIR="../results"`),
so it MUST be run from `experiments/algorithms/`, not the repo root.
```bash
cd experiments/algorithms
python3 GSPH_BASE.py                 # all instances in ../instances/standard
python3 GSPH_BASE.py berlin52        # single instance (name or file.tsp; substring match allowed)
python3 GSPH_BASE.py --profile berlin52   # --profile = N runs (best-of-N), N via GSPH_PROFILE_RUNS (default 10)
GSPH_LIMIT=5 python3 GSPH_BASE.py    # cap number of instances processed
```

### Single strong mode
`GSPH_BASE.py` now always uses the stronger polish: time-driven ILS with
incremental 2-opt reoptimizing only the perturbed region + Or-opt with reversed
insertion. It is designed to get far closer to BKS (often exact) while keeping
runtime adaptive to `n`.
```bash
python3 GSPH_BASE.py a280                        # near/at-BKS, slower than old fast mode
GSPH_QUALITY_TIME_CAP=30 python3 GSPH_BASE.py ...   # per-run time cap (default 30s)
```
Budget scales with `n` (`quality_time_budget()`); for `n>250` the solver bumps
`CAND_KNN` to 40 automatically.

Outputs go to `experiments/results/<instance>/` (`resumen.txt`, `estadisticas_runs.txt`,
and `graficos/` with `ruta_final.png`, `ensamblaje_cuadrantes.png`, `orden_visita.png`,
plus `comparacion_runs.png` + `runs/` under `--profile`) plus a global `results/summary.csv`.
`results/*` is git-ignored (only `.gitkeep` tracked). Plots need `matplotlib`; use the
repo venv (`.venv/bin/python`), the system `python3` here lacks it.

## Key files
- `experiments/algorithms/GSPH_BASE.py` — current/recommended version (2-opt + DLB, Or-opt, ILS,
  kNN + Delaunay candidate sets). Core solver is `gsph_fc()`; tunables are module constants at the
  top (`TIME_BUDGET_S`, `CAND_M`, `CAND_KNN`, `ILS_ITERS`, etc.).
- `experiments/algorithms/GSPH_V2.py` — faster rewrite of the same pipeline: index-based, numba-compiled
  local search (2-opt both directions + Or-opt) and ILS with a local double-bridge that stops on
  stagnation (`GSPH_STALL`, `GSPH_TIME_CAP`, `GSPH_SEED`, `GSPH_TEMP` for worse-move acceptance).
  Moves are logged reversals so rejected kicks are undone without copying the tour. Same CLI/stdout
  as BASE; results go to `experiments/results/v2/`. Falls back to plain Python without numba. BASE
  stays untouched so the paper's numbers remain reproducible.
- `experiments/scripts/benchmark_v2.py` — fast in-process V2 benchmark (no plots) over `standard/` +
  EUC_2D `large/` instances; writes `experiments/results/v2/benchmark_v2.csv`.
- `experiments/algorithms/GSPH_V1.py` — older baseline; keep for comparison, don't develop on it.
- `experiments/scripts/run_all_and_latex.py` — runs the full TSPLIB benchmark and emits LaTeX GAP
  tables. Spawns `GSPH_BASE.py` (or `--algo GSPH_V2`) via subprocess with `cwd=ALGO_DIR`, so run it
  from anywhere. It parses the solver's stdout, so keep the `length=…, time=…s` / `BEST=…` lines.
- `experiments/scripts/compare_runs.py` — builds a grid PNG from `--profile` per-run plots.
- `experiments/scripts/verify_structure.py` — sanity-checks expected layout (note: it lists some
  script names that don't exist; treat its list as aspirational, not authoritative).

## Data / conventions
- Instances are TSPLIB `.tsp` files (`instances/standard/`, `instances/large/`). Parser only reads
  `NODE_COORD_SECTION`; distances are **EUC_2D rounded to integer** (`int(dist + 0.5)`) — this is the
  official TSPLIB metric and how GAP-vs-BKS is computed. `eucl_float()` is used only for fast
  heuristic decisions, not for reported tour length.
- Best-Known Solutions: hardcoded `BKS` dict in `run_all_and_latex.py` and `benchmarks/BKS/BKS_TSP.txt`.
- No test suite, linter config, or CI. `pylint` is installed but unconfigured.

## Gotchas
- `experiments/README.md` references `run_all_instances.py` and `aggregate_resources.py` — these do
  **not** exist. The real entrypoints are the three scripts listed above.
- matplotlib uses the `Agg` backend (no GUI); it writes PNGs directly.
