"""Cliente mínimo para un servidor OSRM (por defecto el local de setup_osrm.sh).

Coordenadas siempre como (lat, lon) hacia afuera; OSRM usa lon,lat en la URL.
"""
import os

import numpy as np
import requests

OSRM_URL = os.getenv('OSRM_URL', 'http://localhost:5001')
_SESION = requests.Session()


def _coords(puntos):
    return ';'.join(f'{lon:.6f},{lat:.6f}' for lat, lon in puntos)


def _get(servicio, puntos, **params):
    url = f'{OSRM_URL}/{servicio}/v1/driving/{_coords(puntos)}'
    r = _SESION.get(url, params=params, timeout=300)
    r.raise_for_status()
    datos = r.json()
    if datos.get('code') != 'Ok':
        raise RuntimeError(f'OSRM {servicio}: {datos.get("code")} {datos.get("message", "")}')
    return datos


def servidor_activo():
    try:
        _get('nearest', [(-33.4372, -70.6506)])
        return True
    except Exception:
        return False


def ajustar_a_calle(lat, lon):
    """Devuelve el punto de calle más cercano (lat, lon) y la distancia de ajuste en metros.

    Excluye autopistas: una parada no puede quedar en la Costanera Norte (las rutas sí las usan).
    """
    w = _get('nearest', [(lat, lon)], number=1, exclude='motorway')['waypoints'][0]
    lon2, lat2 = w['location']
    return (lat2, lon2), w['distance'], w.get('name', '')


def matriz(puntos):
    """Matrices n×n de duración (segundos) y distancia (metros) por la red vial.

    Asimétricas: M[i, j] es ir de i a j. Si OSRM no encuentra camino, deja inf.
    """
    datos = _get('table', puntos, annotations='duration,distance')
    dur = np.array(datos['durations'], dtype=float)
    dist = np.array(datos['distances'], dtype=float)
    dur[~np.isfinite(dur)] = np.inf
    dist[~np.isfinite(dist)] = np.inf
    return dur, dist


def geometria_ruta(puntos_en_orden):
    """Polilínea [(lat, lon), ...] de la ruta que pasa por los puntos en ese orden."""
    datos = _get('route', puntos_en_orden, overview='full', geometries='geojson',
                 continue_straight='false')
    ruta = datos['routes'][0]
    linea = [(lat, lon) for lon, lat in ruta['geometry']['coordinates']]
    return linea, ruta['duration'], ruta['distance']
