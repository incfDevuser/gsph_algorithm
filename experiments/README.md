# TSP Experiments - GSPH Algorithm

Esta carpeta contiene toda la implementación experimental del algoritmo GSPH (Geometric Split and Polish Heuristic) para resolver el problema del viajante (TSP).

## Estructura del Proyecto

```
experiments/
├── algorithms/          # Implementaciones del algoritmo
│   ├── GSPH_BASE.py    # Versión base optimizada (RECOMENDADA)
│   ├── GSPH_V1.py      # Primera versión
│   └── ...             # Otras variantes
│
├── instances/          # Instancias TSP para pruebas
│   ├── standard/       # Instancias estándar (TSPLIB)
│   └── large/          # Instancias grandes
│
├── benchmarks/         # Best Known Solutions (BKS)
│   └── BKS/           # Valores óptimos conocidos
│
├── results/           # Resultados de ejecuciones
│   ├── summary.csv    # Resumen global
│   └── [instancia]/   # Resultados por instancia
│
├── scripts/           # Scripts de utilidad
│   ├── run_all_instances.py    # Ejecutar múltiples instancias
│   └── aggregate_resources.py  # Agregar métricas
│
└── docs/              # Documentación y notas
    ├── apuntes.txt
    └── TEXT.txt
```

## Cómo Ejecutar

### Instalación de Dependencias

```bash
# Desde la raíz del proyecto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ejecución del Algoritmo Base

```bash
# Ejecutar todas las instancias
cd experiments/algorithms
python3 GSPH_BASE.py

# Ejecutar una instancia específica
python3 GSPH_BASE.py berlin52

# Ejecutar con profiling (10 ejecuciones, best-of-10)
python3 GSPH_BASE.py --profile berlin52

# Limitar número de instancias
GSPH_LIMIT=5 python3 GSPH_BASE.py
```

### Ejecutar Múltiples Instancias con Script

```bash
cd experiments/scripts
python3 run_all_instances.py
```

## Resultados

Los resultados se guardan en `experiments/results/`:
- **summary.csv**: Resumen de todas las ejecuciones
- **[instancia]/**: Carpeta por cada instancia con:
  - `resultados.txt`: Estadísticas principales y lista de gráficos generados
  - `grafico_[instancia].png`: Visualización de la ruta (ejecución única)
  - **Con --profile (10 ejecuciones):**
    - `grafico_[instancia]_run_01.png` hasta `run_10.png`: Gráfico de cada ejecución
    - `grafico_[instancia]_run_XX_MEJOR.png`: La mejor ejecución marcada
    - `grafico_[instancia]_MEJOR.png`: Resultado consolidado del mejor run
    - `comparacion_runs.png`: Cuadrícula comparativa (generar con script)
  - `resources.csv`: Uso de memoria/CPU (con --profile)
  - `events.csv`: Timeline de eventos (con --profile)
  - `multiple_runs_stats.txt`: Estadísticas de múltiples runs

## Versiones del Algoritmo

- **GSPH_BASE.py** (⭐ RECOMENDADA): Versión optimizada con:
  - 2-opt con Don't Look Bits (DLB)
  - Or-opt para cadenas de nodos
  - Iterated Local Search (ILS)
  - Candidate sets con kNN + Delaunay
  - Presupuesto de tiempo configurable

- **GSPH_V1.py**: Primera versión funcional

- **Otras variantes**: Versiones experimentales con diferentes enfoques

## Parámetros Principales (GSPH_BASE.py)

```python
TIME_BUDGET_S     = 5.0    # Tiempo para optimización local
CAND_M            = 20     # Candidatos por nodo
CAND_KNN          = 25     # Vecinos cercanos k-NN
USE_DELAUNAY      = True   # Activar triangulación Delaunay
MAX_SWEEPS_2OPT   = 3      # Pasadas de 2-opt
OR_OPT_MAX_L      = 3      # Longitud máxima cadenas Or-opt
ILS_ITERS         = 5      # Iteraciones ILS
```

## Instancias Disponibles

- **Standard**: Instancias TSPLIB clásicas (berlin52, eil101, kroA100, etc.)
- **Large**: Instancias de mayor tamaño para benchmarking

## Scripts de Utilidad

### Ejecutar Múltiples Instancias
```bash
cd experiments/scripts
python3 run_all_instances.py
```

### Agregar Métricas
```bash
python3 aggregate_resources.py
```

### Comparación Visual de Múltiples Runs

Cuando ejecutas con `--profile`, se generan 10 gráficos diferentes (uno por cada ejecución). Para compararlos fácilmente:

```bash
# Instalar Pillow si no lo tienes
pip install Pillow

# Generar cuadrícula comparativa
cd experiments/scripts
python3 compare_runs.py ../results/berlin52

# Esto crea: results/berlin52/comparacion_runs.png
# con todos los gráficos lado a lado en una cuadrícula

# Personalizar número de columnas
python3 compare_runs.py ../results/berlin52 --cols 5
```

## Comparación con BKS

Los valores óptimos conocidos están en `benchmarks/BKS/` para comparar la calidad de las soluciones.

## Notas

- El algoritmo utiliza particionamiento geométrico por cuadrantes
- Optimización local agresiva con múltiples estrategias
- Soporte para profiling de memoria y tiempo
- Resultados reproducibles con múltiples ejecuciones
