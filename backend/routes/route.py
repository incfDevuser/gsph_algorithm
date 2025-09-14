from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Route, Order, OptimizedRoute
from schemas import DepotBase, OrderBase, RouteCreate, RouteResponse
from gsph_algorithm import optimize_route
from typing import List
import json

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=RouteResponse,  summary="Crear ruta", description="Crea una nueva ruta con depósito.")
def create_route(data: RouteCreate, db: Session = Depends(get_db)):
    route = Route(
        depot_id=data.depot.id,
        depot_name=data.depot.name,
        depot_lat=data.depot.lat,
        depot_lng=data.depot.lng
    )
    db.add(route)
    db.commit()
    db.refresh(route)

    return {
        "route_id": route.id,
        "depot": data.depot,
        "orders": []
    }

@router.post("/{route_id}/orders", summary="Agregar órdenes", description="Agrega órdenes a una ruta existente.")
def add_orders(route_id: int, orders: List[OrderBase], db: Session = Depends(get_db)):
    route = db.query(Route).filter(Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")

    for order in orders:
        new_order = Order(
            order_id=order.id,
            lat=order.lat,
            lng=order.lng,
            route_id=route.id
        )
        db.add(new_order)

    db.commit()
    return {"message": f"{len(orders)} órdenes agregadas a la ruta {route_id}."}

@router.get("/{route_id}", summary="Obtener ruta", description="Devuelve el depósito y las órdenes sin optimización.")
def get_raw_route(route_id: int, db: Session = Depends(get_db)):
    route = db.query(Route).filter(Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")

    orders = db.query(Order).filter(Order.route_id == route_id).all()
    return {
        "route_id": route_id,
        "depot": {
            "id": route.depot_id,
            "name": route.depot_name,
            "lat": route.depot_lat,
            "lng": route.depot_lng
        },
        "orders": [{"id": o.order_id, "lat": o.lat, "lng": o.lng} for o in orders]
    }
@router.get("/", summary="Obtener todas las rutas", description="Devuelve una lista de todas las rutas disponibles.")
def get_all_routes(db: Session = Depends(get_db)):
    routes = db.query(Route).all()
    
    result = []
    for route in routes:
        orders_count = db.query(Order).filter(Order.route_id == route.id).count()
        optimized = db.query(OptimizedRoute).filter(OptimizedRoute.route_id == route.id).first() is not None
        
        result.append({
            "route_id": route.id,
            "depot": {
                "id": route.depot_id,
                "name": route.depot_name,
                "lat": route.depot_lat,
                "lng": route.depot_lng
            },
            "orders_count": orders_count,
            "is_optimized": optimized
        })
    
    return result
@router.post("/{route_id}/optimize", summary="Optimizar ruta", description="Ejecuta el algoritmo GSPH y guarda la ruta optimizada.")
def optimize_route_now(route_id: int, method: str = "gsph", db: Session = Depends(get_db)):
    route = db.query(Route).filter(Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Ruta no encontrada")

    existing_opt = db.query(OptimizedRoute).filter(OptimizedRoute.route_id == route_id).first()
    if existing_opt:
        db.delete(existing_opt)
        db.commit()

    orders = db.query(Order).filter(Order.route_id == route_id).all()
    if not orders:
        raise HTTPException(status_code=400, detail="No hay órdenes en esta ruta")

    depot = {
        "id": route.depot_id,
        "name": route.depot_name,
        "lat": route.depot_lat,
        "lng": route.depot_lng
    }

    orders_out = [{"id": o.order_id, "lat": o.lat, "lng": o.lng} for o in orders]
    
    optimized_coords, total_len, order_sequence = optimize_route(depot, orders_out, method=method)

    opt_data = {
        "coordinates": optimized_coords,
        "order_sequence": order_sequence,
        "total_length": total_len,
        "method": method
    }

    opt = OptimizedRoute(
        route_id=route.id,
        coordinates_json=json.dumps(opt_data),
        total_length=total_len
    )
    db.add(opt)
    db.commit()

    return {
        "message": f"Ruta optimizada con {method.upper()} y guardada",
        "route_id": route_id,
        "method": method,
        "optimized_coords": optimized_coords,
        "order_sequence": order_sequence,
        "total_length": total_len
    }

@router.get("/{route_id}/optimized", summary="Obtener ruta optimizada", description="Devuelve la ruta optimizada previamente guardada.")
def get_optimized_route(route_id: int, db: Session = Depends(get_db)):
    opt = db.query(OptimizedRoute).filter(OptimizedRoute.route_id == route_id).first()
    if not opt:
        raise HTTPException(status_code=404, detail="Esta ruta aún no ha sido optimizada")

    coords_data = json.loads(opt.coordinates_json)

    if isinstance(coords_data, list):

        return {
            "route_id": route_id,
            "optimized_coords": coords_data,
            "total_length": opt.total_length
        }
    else:

        return {
            "route_id": route_id,
            "optimized_coords": coords_data.get("coordinates", []),
            "order_sequence": coords_data.get("order_sequence", []),
            "total_length": opt.total_length,
            "method": coords_data.get("method", "gsph")
        }
