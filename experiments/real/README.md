# Rutas reales y entregas con hora exacta

Punto de entrada del proyecto de entregas reales (rama `rutas-reales`). El objetivo es que una flota
de varios repartidores reciba sus rutas sobre calles reales y que a cada cliente se le prometa una
**hora que se cumple**: ±5 minutos, o un rango corto de 1 hora, nunca "de 9 a 18". Todo se valida en
simulación.

El diseño aprobado está en
[`docs/superpowers/specs/2026-10-30-entregas-hora-exacta-design.md`](../../docs/superpowers/specs/2026-10-30-entregas-hora-exacta-design.md).
Léelo antes de escribir código: ahí están las decisiones del usuario y las que quedan al constructor.

## Estado (2026-10-30)

| Pieza | Estado |
|---|---|
| Mapa de Chile + servidor OSRM local (tiempos por calle, sentidos, semáforos) | Listo |
| Solver de ruta para un vehículo sobre calles reales (ATSP) | Listo; en Santiago mejora 1,4–7,3% a OR-Tools |
| Script de ciudad (paradas → matriz → ruta → mapa) | Listo |
| Diseño de entregas con hora exacta | Aprobado |
| Plan de implementación | **Pendiente: siguiente paso** |
| Fases 1–4 del diseño | Sin empezar |

## Lo que ya existe

| Archivo | Qué hace |
|---|---|
| `setup_osrm.sh` | Descarga y procesa el mapa de Chile (si falta) y levanta OSRM en `localhost:5001` (contenedor `osrm-chile`). Variable `SEMAFORO_S` = penalización por semáforo (15 s por defecto; cambiarla requiere `--rebuild`). |
| `osrm_cliente.py` | Ajustar un punto a la calle (sin autopistas), matriz de tiempos y distancias n×n asimétrica, geometría de una ruta. |
| `solver_ruta.py` | `resolver(D, xy, deposito, tiempo, semilla, retorno)`: ATSP con 2-opt orientado (sumas prefijas), Or-opt, ILS double-bridge y embudo multi-arranque. Costos en décimas de segundo. |
| `ciudad.py` | Genera paradas en una zona de Santiago (o lee un CSV), resuelve, compara con OR-Tools y con la ruta optimizada en línea recta, y dibuja `mapa.html`. Salida en `experiments/results/real/<nombre>/`. |

```bash
bash experiments/real/setup_osrm.sh                         # levanta OSRM (el mapa ya está procesado)
.venv/bin/python experiments/real/ciudad.py --zona centro --n 100 --tiempo 10
```

El mapa procesado (~3 GB) vive en `experiments/data/osm/`, que no se versiona.

## Hoja de ruta (fases del diseño)

1. **Mundo simulado + flota sin promesas (política P0, "de 9 a 18").** Tráfico por hora, tiempos de
   entrega variables, ausencias; reparto en zonas por repartidor (partición geométrica tipo GSPH) y
   ruteo con tiempos dependientes de la hora. Es la base de comparación.
2. **Rangos de 1 hora (P1).** Simular cada ruta muchas veces (del orden de 500), prometer el rango
   que se cumple el 95% de las veces, esperar si se llega antes, re-planificar con los rangos como
   ventanas (TSPTW/VRPTW).
3. **Hora exacta en el día (P2).** Unos 45 minutos antes se fija la hora exacta (±5 min) recalculando
   con el estado real; si hay atraso, se reordena lo restante o se avisa al cliente apenas se detecta.
4. **Modo "el cliente elige".** Reservas el día anterior entre los rangos factibles.

Evaluación: comparar P0, P1 y P2 en varios días seguidos (con arrastre de pendientes) y variando la
cantidad de repartidores, para leer "un repartidor más = X entregas más por día".

## Relación con GSPH

En el TSP la partición geométrica solo da un buen punto de partida (el paper del TSP lo muestra con
una ablación). Aquí, con varios repartidores, la partición es la decisión principal: qué zona atiende
cada repartidor. Es el lugar natural para la idea de cuadrantes y pares de frontera.
