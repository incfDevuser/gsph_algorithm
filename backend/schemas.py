from pydantic import BaseModel
from typing import List

class OrderBase(BaseModel):
    id: str
    lat: float
    lng: float

class DepotBase(BaseModel):
    id: str
    name: str
    lat: float
    lng: float

class RouteCreate(BaseModel):
    depot: DepotBase
    orders: List[OrderBase]

class RouteResponse(RouteCreate):
    route_id: int
    
class OptimizedRouteOut(BaseModel):
    optimized_coords: List[List[float]]
    total_length: float
