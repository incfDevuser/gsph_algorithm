from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from gsph_adapter import build_nodes_xy, xy_route_to_indices, build_geojson
from gsph_impl import gsph_fc

app = FastAPI(title="GSPH Delivery API", version="0.1.0")


class Depot(BaseModel):
    id: str
    name: str = "Depot"
    lat: float
    lng: float

class Order(BaseModel):
    id: str
    lat: float
    lng: float

class OptimizeRequest(BaseModel):
    depot: Depot
    orders: List[Order]
    time_budget_s: float = Field(1.5, ge=0, description="Budget de pulido (s)")
    seed: int = 42

@app.post("/optimize")
def optimize(req: OptimizeRequest) -> Dict[str, Any]:
    built = build_nodes_xy(req.depot.model_dump(), [o.model_dump() for o in req.orders])
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

    geojson = build_geojson(req.depot.model_dump(), [o.model_dump() for o in req.orders],
                            idx_tour, nodes_ll, connections, nodes_xy)

    return {
        "metrics": {
            "total_distance_km": dist_km,
            "nodes": len(nodes_ll),
        },
        "solution": geojson
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)