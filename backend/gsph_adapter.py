from typing import List, Tuple, Dict, Any
import math
import json
import numpy as np
from scipy.spatial import cKDTree

def project_latlng_to_xy_m(lat: float, lng: float, lat0: float, lng0: float) -> Tuple[float, float]:
    R = 6371000.0  # m
    lat_r = math.radians(lat)
    lng_r = math.radians(lng)
    lat0_r = math.radians(lat0)
    lng0_r = math.radians(lng0)
    x = R * (lng_r - lng0_r) * math.cos((lat_r + lat0_r) / 2.0)
    y = R * (lat_r - lat0_r)
    return (x, y)

def build_nodes_xy(depot: Dict[str, float], orders: List[Dict[str, float]]) -> Dict[str, Any]:
    lat0, lng0 = depot["lat"], depot["lng"]
    nodes_ll = [(depot["lat"], depot["lng"])] + [(o["lat"], o["lng"]) for o in orders]
    nodes_xy = [project_latlng_to_xy_m(lat, lng, lat0, lng0) for (lat, lng) in nodes_ll]
    return {
        "nodes_ll": nodes_ll,  
        "nodes_xy": nodes_xy  
    }

def xy_route_to_indices(route_xy: List[Tuple[float, float]], nodes_xy: List[Tuple[float, float]]) -> List[int]:
    P = np.array(nodes_xy, dtype=float)
    tree = cKDTree(P)
    idxs = []
    for p in route_xy:
        d, i = tree.query(np.array(p, dtype=float))
        idxs.append(int(i))
    dedup = []
    for i in idxs:
        if not dedup or dedup[-1] != i:
            dedup.append(i)
    return dedup

def build_geojson(depot: Dict[str, Any], orders: List[Dict[str, Any]], idx_tour: List[int],
                  nodes_ll: List[Tuple[float,float]], connections_xy: List[Tuple[Tuple[float,float], Tuple[float,float]]],
                  nodes_xy: List[Tuple[float,float]]) -> Dict[str, Any]:
    features = []
    dep_lat, dep_lng = nodes_ll[0]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [dep_lng, dep_lat]},
        "properties": {"id": depot["id"], "name": depot.get("name", "Depot"), "type": "depot"}
    })
    for i, o in enumerate(orders, start=1):
        lat, lng = nodes_ll[i]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {"id": o["id"], "type": "order"}
        })

    coords = []
    for idx in idx_tour:
        lat, lng = nodes_ll[idx]
        coords.append([lng, lat])
    if idx_tour and idx_tour[0] != idx_tour[-1]:
        lat0, lng0 = nodes_ll[idx_tour[0]]
        coords.append([lng0, lat0])

    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"layer": "tour", "vehicle": "V1"}
    })

    if connections_xy:
        conn_coords = []
        for (a_xy, b_xy) in connections_xy:
            tree = cKDTree(np.array(nodes_xy, dtype=float))
            _, ia = tree.query(np.array(a_xy, dtype=float))
            _, ib = tree.query(np.array(b_xy, dtype=float))
            la, lnga = nodes_ll[int(ia)]
            lb, lngb = nodes_ll[int(ib)]
            conn_coords.append([[lnga, la], [lngb, lb]])

        features.append({
            "type": "Feature",
            "geometry": {"type": "MultiLineString", "coordinates": conn_coords},
            "properties": {"layer": "connections"}
        })

    return {"type": "FeatureCollection", "features": features}
