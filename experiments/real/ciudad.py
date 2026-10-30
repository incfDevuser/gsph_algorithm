"""Ruta real en una ciudad: paradas -> matriz de tiempos OSRM -> solver -> mapa.

Requiere el servidor OSRM local (bash experiments/real/setup_osrm.sh).

Uso (desde cualquier carpeta, con el venv del repo):
  python experiments/real/ciudad.py --n 100                       # 100 paradas al azar en Santiago Centro
  python experiments/real/ciudad.py --zona providencia --n 200 --tiempo 20
  python experiments/real/ciudad.py --csv paradas.csv             # columnas lat,lon[,nombre]; la 1a fila es el depósito
  python experiments/real/ciudad.py --n 150 --sin-retorno         # ruta abierta (no vuelve al depósito)
  python experiments/real/ciudad.py --instancia ../results/real/centro_n100_s0/instancia.npz

Compara contra OR-Tools (GLS, mismo tiempo) y contra la ruta optimizada en línea
recta y medida en calles reales. Salida en experiments/results/real/<nombre>/.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import osrm_cliente as osrm  # noqa: E402
import solver_ruta  # noqa: E402

RESULTADOS = os.path.join(AQUI, '..', 'results', 'real')

# (lat_min, lat_max, lon_min, lon_max)
ZONAS = {
    'centro':      (-33.470, -33.425, -70.680, -70.630),
    'providencia': (-33.445, -33.410, -70.630, -70.580),
    'nunoa':       (-33.475, -33.440, -70.625, -70.575),
    'santiago':    (-33.580, -33.370, -70.760, -70.540),   # Gran Santiago
    'concepcion':  (-36.850, -36.790, -73.090, -73.020),
}
MAX_AJUSTE_M = 80   # descarta puntos que caen a más de esto de una calle (cerros, parques)


def proyectar(puntos):
    """lat/lon -> metros (equirectangular local; suficiente a escala de ciudad)."""
    P = np.asarray(puntos, float)
    lat0 = math.radians(P[:, 0].mean())
    x = np.radians(P[:, 1]) * 6371000.0 * math.cos(lat0)
    y = np.radians(P[:, 0]) * 6371000.0
    return np.column_stack([x, y])


def generar_paradas(zona, n, semilla):
    lat0, lat1, lon0, lon1 = ZONAS[zona]
    rng = np.random.default_rng(semilla)
    centro, _, _ = osrm.ajustar_a_calle((lat0 + lat1) / 2, (lon0 + lon1) / 2)
    puntos, nombres, vistos = [centro], ['Depósito'], {(round(centro[0], 5), round(centro[1], 5))}
    intentos = 0
    while len(puntos) < n + 1 and intentos < 20 * n:
        intentos += 1
        p, dist, calle = osrm.ajustar_a_calle(rng.uniform(lat0, lat1), rng.uniform(lon0, lon1))
        clave = (round(p[0], 5), round(p[1], 5))
        if dist > MAX_AJUSTE_M or clave in vistos:
            continue
        vistos.add(clave)
        puntos.append(p)
        nombres.append(calle or f'Parada {len(puntos) - 1}')
    return puntos, nombres


def leer_csv(ruta):
    puntos, nombres = [], []
    with open(ruta, newline='', encoding='utf-8') as f:
        for i, fila in enumerate(csv.DictReader(f)):
            puntos.append((float(fila['lat']), float(fila['lon'])))
            nombres.append(fila.get('nombre') or ('Depósito' if i == 0 else f'Parada {i}'))
    return puntos, nombres


def quitar_inalcanzables(puntos, nombres, dur, dist):
    """Saca paradas sin camino de ida o vuelta (islas de la red), nunca el depósito."""
    ok = np.isfinite(dur).all(axis=1) & np.isfinite(dur).all(axis=0)
    ok[0] = True
    if not ok.all():
        print(f'  Aviso: {int((~ok).sum())} parada(s) sin camino, se descartan')
        idx = np.flatnonzero(ok)
        puntos = [puntos[i] for i in idx]
        nombres = [nombres[i] for i in idx]
        dur, dist = dur[np.ix_(idx, idx)], dist[np.ix_(idx, idx)]
    return puntos, nombres, dur, dist


def ortools(D, tiempo, retorno):
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None
    n = len(D)
    Dm = D.copy()
    if not retorno:
        Dm[:, 0] = 0
    man = pywrapcp.RoutingIndexManager(n, 1, 0)
    rt = pywrapcp.RoutingModel(man)
    cb = rt.RegisterTransitMatrix(Dm.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(tiempo * 1000))
    t0 = time.monotonic()
    sol = rt.SolveWithParameters(p)
    orden, idx = [], rt.Start(0)
    while not rt.IsEnd(idx):
        orden.append(man.IndexToNode(idx))
        idx = sol.Value(rt.NextVar(idx))
    return {'orden': orden, 'tiempo_s': time.monotonic() - t0}


def dibujar_mapa(puntos, nombres, rutas, salida):
    try:
        import folium
    except ImportError:
        print('  (folium no instalado: sin mapa)')
        return
    m = folium.Map(location=np.mean(puntos, axis=0).tolist(), zoom_start=14, tiles='OpenStreetMap')
    for k, (etiqueta, orden, color, visible, retorno) in enumerate(rutas):
        capa = folium.FeatureGroup(name=etiqueta, show=visible)
        secuencia = [puntos[i] for i in orden] + ([puntos[orden[0]]] if retorno else [])
        linea, _, _ = osrm.geometria_ruta(secuencia)
        folium.PolyLine(linea, color=color, weight=4 if k == 0 else 3, opacity=0.85,
                        dash_array=None if k == 0 else '6 6').add_to(capa)
        capa.add_to(m)
    orden0 = rutas[0][1]
    for paso, i in enumerate(orden0):
        lat, lon = puntos[i]
        if paso == 0:
            folium.Marker((lat, lon), tooltip='Depósito', icon=folium.Icon(color='red', icon='home')).add_to(m)
        else:
            folium.Marker((lat, lon), tooltip=f'{paso}. {nombres[i]}', icon=folium.DivIcon(html=(
                '<div style="background:#1f4e9c;color:#fff;border-radius:50%;width:20px;height:20px;'
                f'font:600 10px sans-serif;display:flex;align-items:center;justify-content:center;'
                f'transform:translate(-10px,-10px)">{paso}</div>'))).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(salida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zona', default='centro', choices=sorted(ZONAS))
    ap.add_argument('--n', type=int, default=100, help='paradas (sin contar el depósito)')
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--csv', help='archivo lat,lon[,nombre]; la primera fila es el depósito')
    ap.add_argument('--instancia', help='instancia.npz guardada por una corrida anterior')
    ap.add_argument('--tiempo', type=float, default=10.0, help='segundos por solver')
    ap.add_argument('--sin-retorno', action='store_true', help='no volver al depósito')
    ap.add_argument('--sin-ortools', action='store_true')
    ap.add_argument('--sin-mapa', action='store_true')
    args = ap.parse_args()
    retorno = not args.sin_retorno

    if args.instancia:
        z = np.load(args.instancia, allow_pickle=False)
        puntos, nombres = [tuple(p) for p in z['puntos']], list(z['nombres'])
        dur, dist = z['dur'], z['dist']
        nombre = os.path.basename(os.path.dirname(os.path.abspath(args.instancia)))
    else:
        if not osrm.servidor_activo():
            sys.exit(f'No hay servidor OSRM en {osrm.OSRM_URL}. Corre: bash experiments/real/setup_osrm.sh')
        if args.csv:
            puntos, nombres = leer_csv(args.csv)
            nombre = os.path.splitext(os.path.basename(args.csv))[0]
        else:
            print(f'Generando {args.n} paradas en {args.zona}...')
            puntos, nombres = generar_paradas(args.zona, args.n, args.semilla)
            nombre = f'{args.zona}_n{args.n}_s{args.semilla}'
        t0 = time.monotonic()
        dur, dist = osrm.matriz(puntos)
        print(f'Matriz {len(puntos)}×{len(puntos)} de OSRM en {time.monotonic() - t0:.1f}s')
        puntos, nombres, dur, dist = quitar_inalcanzables(puntos, nombres, dur, dist)
    if not retorno:
        nombre += '_abierta'

    n = len(puntos)
    D = np.rint(dur * 10).astype(np.int64)          # décimas de segundo
    asim = np.abs(dur - dur.T)[np.triu_indices(n, 1)]
    xy = proyectar(puntos)
    recta = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    desvio = (dist / np.maximum(recta, 1.0))[~np.eye(n, dtype=bool) & (recta > 200)]

    print(f'{n} nodos (depósito + {n - 1} paradas). Asimetría ida/vuelta: mediana {np.median(asim):.0f}s, '
          f'p90 {np.percentile(asim, 90):.0f}s. Calle/recta: mediana {np.median(desvio):.2f}x')

    solver_ruta.calentar()
    res = solver_ruta.resolver(D, xy=xy, tiempo=args.tiempo, semilla=args.semilla, retorno=retorno)
    filas = [('Solver ruta (ATSP)', res['orden'], res['tiempo_s'])]

    # Misma ruta pero optimizada en línea recta y medida en calles: cuánto cuesta ignorar la red.
    Dr = np.rint(recta).astype(np.int64)
    rr = solver_ruta.resolver(Dr, xy=xy, tiempo=min(args.tiempo, 5.0), semilla=args.semilla, retorno=retorno)
    filas.append(('Optimizada en línea recta', rr['orden'], rr['tiempo_s']))

    if not args.sin_ortools:
        ro = ortools(D, args.tiempo, retorno)
        if ro:
            filas.append(('OR-Tools GLS', ro['orden'], ro['tiempo_s']))

    base = solver_ruta.costo_ruta(D, res['orden'], retorno)
    lineas = [f'Instancia: {nombre}  ({n} nodos, retorno={"sí" if retorno else "no"}, tiempo={args.tiempo:g}s)',
              f'Arranque ganador: {res["arranque"]}  kicks ILS: {res["kicks"]}',
              f'Arranques (tras búsqueda local, décimas de s): {res["arranques"]}', '',
              f'{"Método":28s} {"min":>8s} {"km":>8s} {"vs solver":>10s} {"t(s)":>6s}']
    for etiqueta, orden, t in filas:
        c = solver_ruta.costo_ruta(D, orden, retorno)
        km = solver_ruta.costo_ruta(dist, orden, retorno) / 1000
        lineas.append(f'{etiqueta:28s} {c / 600:8.1f} {km:8.2f} {(c - base) / base * 100:+9.2f}% {t:6.1f}')
    print('\n'.join(lineas))

    outdir = os.path.join(RESULTADOS, nombre)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'resumen.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')
    with open(os.path.join(outdir, 'orden.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['paso', 'nodo', 'lat', 'lon', 'nombre', 'llegada_min'])
        acum = 0.0
        orden = res['orden']
        for paso, i in enumerate(orden):
            if paso:
                acum += dur[orden[paso - 1], i]
            w.writerow([paso, i, f'{puntos[i][0]:.6f}', f'{puntos[i][1]:.6f}', nombres[i], f'{acum / 60:.1f}'])
    if not args.instancia:
        np.savez_compressed(os.path.join(outdir, 'instancia.npz'), puntos=np.array(puntos),
                            nombres=np.array(nombres), dur=dur, dist=dist)
    if not args.sin_mapa and osrm.servidor_activo():
        colores = ['#1f4e9c', '#888888', '#e07b00']
        rutas = [(e, o, colores[k], k == 0, retorno) for k, (e, o, _) in enumerate(filas)]
        dibujar_mapa(puntos, nombres, rutas, os.path.join(outdir, 'mapa.html'))
    print(f'Resultados en {os.path.relpath(outdir)}')


if __name__ == '__main__':
    main()
