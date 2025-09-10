from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from gsph_adapter import build_nodes_xy, xy_route_to_indices, build_geojson
from gsph_impl import gsph_fc
import uuid
from fastapi.middleware.cors import CORSMiddleware
import datetime

app = FastAPI(title="GSPH Delivery API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routes_db = {}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "GSPH Delivery API is running"}


class Location(BaseModel):
    lat: float
    lng: float


class Order(BaseModel):
    id: str
    name: Optional[str] = None
    location: Location


class Depot(BaseModel):
    id: str
    name: str = "Depot"
    location: Location


class OptimizeRequest(BaseModel):
    depot: Depot
    orders: List[Order]
    time_budget_s: float = Field(1.5, ge=0, description="Budget de pulido (s)")
    seed: int = 42


class RouteBase(BaseModel):
    name: str
    depot: Depot
    orders: List[Order] = []


class Route(RouteBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    optimized_route: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class RouteUpdate(BaseModel):
    name: Optional[str] = None
    depot: Optional[Depot] = None

# API para gestión de rutas
@app.post("/routes", response_model=Route)
def create_route(route: RouteBase):
    """Crear una nueva ruta con un depósito y órdenes opcionales"""
    new_route = Route(
        id=str(uuid.uuid4()),
        name=route.name,
        depot=route.depot,
        orders=route.orders,
        created_at=datetime.datetime.now().isoformat()
    )
    routes_db[new_route.id] = new_route.model_dump()
    return new_route


@app.get("/routes", response_model=List[Route])
def get_all_routes():
    """Obtener todas las rutas existentes"""
    return list(routes_db.values())


@app.get("/routes/{route_id}", response_model=Route)
def get_route(route_id: str):
    """Obtener una ruta específica por su ID"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    return routes_db[route_id]


@app.put("/routes/{route_id}", response_model=Route)
def update_route(route_id: str, route_update: RouteUpdate):
    """Actualizar una ruta existente"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    current_route = routes_db[route_id]
    
    # Actualizar solo los campos proporcionados
    if route_update.name is not None:
        current_route["name"] = route_update.name
    if route_update.depot is not None:
        current_route["depot"] = route_update.depot.model_dump()
    
    routes_db[route_id] = current_route
    return current_route


@app.delete("/routes/{route_id}")
def delete_route(route_id: str):
    """Eliminar una ruta existente"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    del routes_db[route_id]
    return {"message": "Ruta eliminada correctamente"}


@app.post("/routes/{route_id}/orders", response_model=Route)
def add_order_to_route(route_id: str, order: Order):
    """Agregar una orden a una ruta existente"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    current_route = routes_db[route_id]
    current_route["orders"].append(order.model_dump())
    
    # Resetear la ruta optimizada si había una
    if "optimized_route" in current_route and current_route["optimized_route"]:
        current_route["optimized_route"] = None
    
    routes_db[route_id] = current_route
    return current_route


@app.delete("/routes/{route_id}/orders/{order_id}", response_model=Route)
def remove_order_from_route(route_id: str, order_id: str):
    """Eliminar una orden de una ruta existente"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    current_route = routes_db[route_id]
    
    # Buscar y eliminar la orden
    found = False
    orders = current_route["orders"]
    for i, order in enumerate(orders):
        if order["id"] == order_id:
            orders.pop(i)
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Orden no encontrada en esta ruta")
    
    # Resetear la ruta optimizada si había una
    if "optimized_route" in current_route and current_route["optimized_route"]:
        current_route["optimized_route"] = None
    
    routes_db[route_id] = current_route
    return current_route

# Endpoint para optimizar una ruta directamente con un request
@app.post("/optimize")
def optimize(req: OptimizeRequest) -> Dict[str, Any]:
    # Adaptar el formato del depósito
    depot_data = {
        "id": req.depot.id,
        "name": req.depot.name,
        "lat": req.depot.location.lat,
        "lng": req.depot.location.lng
    }
    
    # Adaptar el formato de las órdenes
    orders_data = [
        {
            "id": order.id,
            "name": order.name if order.name else f"Orden {i+1}",
            "lat": order.location.lat,
            "lng": order.location.lng
        } for i, order in enumerate(req.orders)
    ]
    
    built = build_nodes_xy(depot_data, orders_data)
    nodes_ll = built["nodes_ll"]
    nodes_xy = built["nodes_xy"]

    from gsph_impl import GSPHConfig
    cfg = GSPHConfig(
        time_budget_s=req.time_budget_s,
        seed=req.seed
    )

    routes_out, connections, total_len_xy, xmid, ymid = gsph_fc(nodes_xy, cfg)

    qall_xy = routes_out.get("QALL", [])
    idx_tour = xy_route_to_indices(qall_xy, nodes_xy)

    distance_units = "units_xy" 

    dist_km = 0.0
    if idx_tour:
        total_m = 0.0
        for i in range(len(idx_tour)-1):
            a = nodes_xy[idx_tour[i]]
            b = nodes_xy[idx_tour[i+1]]
            total_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        a = nodes_xy[idx_tour[-1]]
        b = nodes_xy[idx_tour[0]]
        total_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        dist_km = round(total_m / 1000.0, 3)

    geojson = build_geojson(depot_data, orders_data, idx_tour, nodes_ll, connections, nodes_xy)

    result = {
        "metrics": {
            "total_distance_km": dist_km,
            "nodes": len(nodes_ll),
        },
        "solution": geojson
    }

    return result


@app.post("/load-test-data")
def load_test_data():
    """Cargar datos de prueba para facilitar el desarrollo"""
    routes_db.clear()
    
    depot = Depot(
        id="depot-1",
        name="Depósito Central",
        location=Location(lat=-33.4489, lng=-70.6693)
    )
    orders = [
        Order(id="order-1", name="Providencia", location=Location(lat=-33.4314, lng=-70.6093)),
        Order(id="order-2", name="Las Condes", location=Location(lat=-33.4145, lng=-70.5836)),
        Order(id="order-3", name="Vitacura", location=Location(lat=-33.3923, lng=-70.5705)),
        Order(id="order-4", name="Lo Barnechea", location=Location(lat=-33.3571, lng=-70.5126)),
        Order(id="order-5", name="La Florida", location=Location(lat=-33.5422, lng=-70.5995)),
    ]
    new_route = Route(
        id="route-test-1",
        name="Ruta de Prueba Santiago",
        depot=depot,
        orders=orders,
        created_at=datetime.datetime.now().isoformat()
    )
    
    routes_db[new_route.id] = new_route.model_dump()
    
    depot2 = Depot(
        id="depot-2",
        name="Centro de Distribución",
        location=Location(lat=-33.5129, lng=-70.7538) 
    )
    
    orders2 = [
        Order(id="order-6", name="Pudahuel", location=Location(lat=-33.4396, lng=-70.7680)),
        Order(id="order-7", name="Cerrillos", location=Location(lat=-33.4957, lng=-70.7079)),
        Order(id="order-8", name="Quilicura", location=Location(lat=-33.3490, lng=-70.7295)),
        Order(id="order-9", name="Renca", location=Location(lat=-33.4080, lng=-70.7281)),
    ]
    
    new_route2 = Route(
        id="route-test-2",
        name="Ruta Zona Poniente",
        depot=depot2,
        orders=orders2,
        created_at=datetime.datetime.now().isoformat()
    )
    
    routes_db[new_route2.id] = new_route2.model_dump()
    
    return {
        "message": "Datos de prueba cargados correctamente",
        "routes": [new_route.model_dump(), new_route2.model_dump()]
    }


# Endpoint para optimizar una ruta existente
@app.post("/routes/{route_id}/optimize", response_model=Route)
def optimize_route(route_id: str, time_budget_s: float = 1.5, seed: int = 42):
    """Optimizar una ruta existente"""
    if route_id not in routes_db:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")
    
    current_route = routes_db[route_id]
    
    # Verificar que hay órdenes para optimizar
    if not current_route["orders"]:
        raise HTTPException(status_code=400, detail="La ruta no tiene órdenes para optimizar")
    
    # Crear el request de optimización
    depot_data = {
        "id": current_route["depot"]["id"],
        "name": current_route["depot"]["name"],
        "lat": current_route["depot"]["location"]["lat"],
        "lng": current_route["depot"]["location"]["lng"]
    }
    
    orders_data = [
        {
            "id": order["id"],
            "name": order.get("name", f"Orden {i+1}"),
            "lat": order["location"]["lat"],
            "lng": order["location"]["lng"]
        } for i, order in enumerate(current_route["orders"])
    ]
    
    # Ejecutar la optimización
    built = build_nodes_xy(depot_data, orders_data)
    nodes_ll = built["nodes_ll"]
    nodes_xy = built["nodes_xy"]

    from gsph_impl import GSPHConfig
    cfg = GSPHConfig(
        time_budget_s=time_budget_s,
        seed=seed
    )

    routes_out, connections, total_len_xy, xmid, ymid = gsph_fc(nodes_xy, cfg)

    qall_xy = routes_out.get("QALL", [])
    idx_tour = xy_route_to_indices(qall_xy, nodes_xy)

    dist_km = 0.0
    if idx_tour:
        total_m = 0.0
        for i in range(len(idx_tour)-1):
            a = nodes_xy[idx_tour[i]]
            b = nodes_xy[idx_tour[i+1]]
            total_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        a = nodes_xy[idx_tour[-1]]
        b = nodes_xy[idx_tour[0]]
        total_m += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        dist_km = round(total_m / 1000.0, 3)

    geojson = build_geojson(depot_data, orders_data, idx_tour, nodes_ll, connections, nodes_xy)

    result = {
        "metrics": {
            "total_distance_km": dist_km,
            "nodes": len(nodes_ll),
        },
        "solution": geojson
    }
    
    # Guardar el resultado de la optimización en la ruta
    current_route["optimized_route"] = result
    routes_db[route_id] = current_route
    
    return current_route
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)