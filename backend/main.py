from fastapi import FastAPI
from routes.route import router as route_router

app = FastAPI(title="GSPH API")

app.include_router(route_router, prefix="/routes", tags=["Routes"])
