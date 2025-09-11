from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.route import router as route_router

app = FastAPI(
    title="API de Ruteo Logístico",
    description="Sistema para crear, gestionar y optimizar rutas logísticas usando GSPH.",
    version="1.0.0"
)
origins = [
    "http://localhost:3000",        
    "http://127.0.0.1:3000",
    "http://localhost:19006",  
    "http://localhost:5173",     
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],            
)
app.include_router(route_router, prefix="/routes", tags=["Rutas"])
