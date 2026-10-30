# AGENTS.md

Research repo for the **GSPH** (Geometric Split and Polish Heuristic) TSP algorithm.
Two parts: Python experiment harness under `experiments/`, and a LaTeX paper at the root
(`samplepaper_v3.tex` is the current paper and `samplepaper_v3_es.tex` its Spanish translation, kept in
sync; `samplepaper_v2*.tex`, `samplepaper.tex` and `test.tex` are older versions;
all cite `bibliografia.bib`). Code comments and docs are in Spanish. `REPRODUCIR.md` lists the command
behind every table and figure.

## Active work on this branch (`rutas-reales`)
Real-delivery project (not the TSPLIB paper). The branch already includes all paper work (`main` +
the v3 from `revision-paper`), so it merges cleanly into `main`; keep paper edits on
`revision-paper`/`main` and this branch's commits about `experiments/real/`. Goal: several drivers, real-street routes, exact promised delivery times
(±5 min or a 1-hour window), in simulation. Approved design:
`docs/superpowers/specs/2026-10-30-entregas-hora-exacta-design.md`. Nothing of it is implemented yet;
next step is the implementation plan for its 4 phases. Existing pieces and roadmap:
`experiments/real/README.md`. Start OSRM first: `bash experiments/real/setup_osrm.sh` (port 5001).

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
  is the paper's "previous implementation" and only received a crash fix in `double_bridge_tracked`.
- Paper revision scripts in `experiments/scripts/`: `ablacion_construccion.py` (construction ablation),
  `sensibilidad_parametros.py`, `comparar_lkh.py` (LKH binary in `experiments/data/lkh/` or LKH-3 via
  `elkai`), `remedir_v1.py`, `convergencia.py` (GAP vs time figure), `held_karp.py`, and
  `tablas_revision.py` (CSV → LaTeX). Details in CLAUDE.md.
- `experiments/scripts/benchmark_v2.py` — fast in-process V2 benchmark (no plots) over `standard/` +
  EUC_2D `large/` instances; writes `experiments/results/v2/benchmark_v2.csv`.
- `experiments/algorithms/GSPH_V1.py` — older baseline; keep for comparison, don't develop on it.
- `experiments/scripts/run_all_and_latex.py` — runs the full TSPLIB benchmark and emits LaTeX GAP
  tables. Spawns `GSPH_BASE.py` (or `--algo GSPH_V2`) via subprocess with `cwd=ALGO_DIR`, so run it
  from anywhere. It parses the solver's stdout, so keep the `length=…, time=…s` / `BEST=…` lines.
- `experiments/scripts/compare_runs.py` — builds a grid PNG from `--profile` per-run plots.
- `experiments/scripts/verify_structure.py` — sanity-checks expected layout (note: it lists some
  script names that don't exist; treat its list as aspirational, not authoritative).

## Real road routes (`experiments/real/`)
Separate from the TSPLIB pipeline: stops on a real street network, cost = OSRM travel time.
- `setup_osrm.sh` downloads `chile-latest.osm.pbf` (Geofabrik, ~350 MB) into the git-ignored
  `experiments/data/osm/`, processes it with the official `osrm-backend:v6.0.0` docker image (~1 min)
  and runs container `osrm-chile` on **port 5001** (5000 is macOS AirPlay). `osrm_cliente.py` reads
  `OSRM_URL`. OSRM v6's `car.lua` has no signal penalty (falls back to 2 s); the script injects
  `traffic_signal_penalty` (env `SEMAFORO_S`, default 15 s; change it with `--rebuild`).
- `solver_ruta.py`: ATSP solver (`resolver(D, xy, deposito, tiempo, semilla, retorno)`), integer
  costs in tenths of a second. Same ideas as V2 (candidate lists, queue local search, local
  double-bridge ILS with temperature, multi-start funnel incl. a Hilbert/quadrant start), but rewritten
  for asymmetric costs: the 2-opt never reverses the complement (in ATSP that is another tour), it
  evaluates both reconnections of each edge pair, and the inner reversed-arc cost comes from forward/
  backward prefix sums `F`/`B`, recomputed after every applied move (O(n), fine for ≤ a few thousand
  stops). Rejected kicks restore a full copy (no journal). `retorno=False` = open route via
  `D[:, depot] = 0`. Validated vs Held-Karp (n ≤ 12, 0 gap).
- `ciudad.py`: random stops in a preset zone (`ZONAS`) snapped with `/nearest?exclude=motorway` (else
  stops land on Costanera Norte), or `--csv lat,lon[,nombre]` (row 1 = depot), or `--instancia
  instancia.npz` to rerun without OSRM. Compares against OR-Tools GLS (same time) and the
  straight-line-optimized tour measured on streets; writes `experiments/results/real/<name>/`
  (`resumen.txt`, `orden.csv`, `instancia.npz`, `mapa.html` via folium + OSRM route geometry).

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
