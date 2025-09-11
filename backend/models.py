from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class Route(Base):
    __tablename__ = "routes"
    id = Column(Integer, primary_key=True, index=True)
    depot_id = Column(String)
    depot_name = Column(String)
    depot_lat = Column(Float)
    depot_lng = Column(Float)
    orders = relationship("Order", back_populates="route")
    optimized_route = relationship("OptimizedRoute", back_populates="route", uselist=False)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String)
    lat = Column(Float)
    lng = Column(Float)
    route_id = Column(Integer, ForeignKey("routes.id"))
    route = relationship("Route", back_populates="orders")

class OptimizedRoute(Base):
    __tablename__ = "optimized_routes"
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), unique=True)
    coordinates_json = Column(Text)
    total_length = Column(Float)
    route = relationship("Route", back_populates="optimized_route")
