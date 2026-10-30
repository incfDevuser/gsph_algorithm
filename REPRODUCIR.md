# Reproducir los resultados del paper GSPH-FC

> **English summary.** Every table and figure of the paper is produced by one script in
> `experiments/scripts/`. Create the virtual environment (Python ≥ 3.11, `pip install -r
> requirements.txt`), then run the commands below from the repository root. All runs are seeded
> (run *k* uses seed 1000 + *k*), every tour is validated as a permutation and its length is
> recomputed from the coordinates, and results are written as CSV under `experiments/results/`.
> Wall-clock times depend on the machine; the paper used an Apple M4 Pro (single thread).

Todos los comandos se corren desde la raíz del repositorio, con el entorno virtual activo.

## 1. Entorno

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- `numba` es opcional pero necesario para los tiempos del paper (sin él, el mismo código corre en
  Python puro, mucho más lento). La primera ejecución compila los núcleos (unos segundos).
- Para la comparación con LKH: `pip install elkai` (trae LKH-3.0.8). En Python 3.14 hay que
  compilarlo con `PIP_CONSTRAINT` apuntando a un archivo que contenga `scikit-build-core<0.10`.
  Si existe el binario oficial en `experiments/data/lkh/LKH-2.0.10/LKH`, se usa ese.

## 2. Tablas y figuras

| Resultado del paper | Comando | Salida | Tiempo aprox.* |
|---|---|---|---|
| Tablas por instancia y resumen por tamaño | `python experiments/scripts/benchmark_v2.py --runs 10 --large-runs 3` | `experiments/results/v2/benchmark_v2.csv` | ~1,5 h |
| Óptimos de `eil10`/`eil20` | `python experiments/scripts/held_karp.py eil10 eil20` | pantalla | ~10 s |
| Ablación de la construcción | `python experiments/scripts/ablacion_construccion.py --max-n 2000 --runs 5 --workers 3` | `experiments/results/ablacion/` | ~15 min |
| Curvas de convergencia (figura) | `python experiments/scripts/convergencia.py --workers 2` | `experiments/results/convergencia/` | ~5 min |
| Ablación de la temperatura | `GSPH_TEMP=0 python experiments/scripts/benchmark_v2.py --runs 5 --no-large --only kroE100 lin318 pcb442 rat783 pr1002 d1655 --out experiments/results/v2/temp0.csv` (y lo mismo sin `GSPH_TEMP` para T = 0,2) | `experiments/results/v2/temp0.csv` | ~2 min |
| Sensibilidad de parámetros | `python experiments/scripts/sensibilidad_parametros.py --amplio --runs 5 --workers 4` | `experiments/results/sensibilidad/sens_amplio_*.csv` | ~10 min |
| Comparación con LKH-3 | `python experiments/scripts/comparar_lkh.py --max-n 2000 --runs 3 --workers 4` | `experiments/results/lkh/` | ~5 min |
| Implementación anterior | `python experiments/scripts/remedir_v1.py --workers 4` | `experiments/results/v1_remedido/` | ~45 min |
| Tablas LaTeX desde los CSV | `python experiments/scripts/tablas_revision.py` | `experiments/results/tablas_revision.tex` | segundos |

\* Con varios procesos en paralelo. Los tiempos medidos en paralelo solo se comparan dentro de una
misma tabla; para tiempos absolutos, correr con `--workers 1` y sin otros procesos.

## 3. Detalles del protocolo

- **Instancias:** `experiments/instances/standard/` (74 archivos con n ≤ 6.000) y las 6 instancias
  `EUC_2D` de `experiments/instances/large/`. Se excluyen de los conteos `eil10`, `eil20`
  (subconjuntos de `eil51`) y `linhp318` (su BKS es de una variante con arista fija). Los `pla*`
  (`CEIL_2D`) no se usan.
- **Métrica:** TSPLIB `EUC_2D`, distancia redondeada al entero más cercano.
- **BKS:** `experiments/benchmarks/BKS/BKS_TSP.txt`.
- **Semillas:** el run *k* usa la semilla `1000 + k` (`--seed` cambia la base). La implementación
  anterior (`GSPH_BASE.py`) no usa semilla, así que sus resultados varían levemente entre corridas.
- **Parámetros del método:** constantes al inicio de `experiments/algorithms/GSPH_V2.py`; las
  variables de entorno `GSPH_TEMP`, `GSPH_STALL`, `GSPH_TIME_CAP` y `GSPH_SEED` los sobrescriben.
