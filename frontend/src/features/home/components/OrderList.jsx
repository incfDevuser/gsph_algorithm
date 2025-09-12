import React, { useEffect, useState } from "react";
import OrdenCard from "./OrdenCard";
import { Search, FileDown, Filter, Plus, Map } from "lucide-react";
import { useGSPHStore } from "../../../store/GSPHStore";

const OrderList = ({ depot, orders, optimized }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const { loading, fetchAllRoutes, fetchRoute } = useGSPHStore();
  const [availableRoutes, setAvailableRoutes] = useState([]);
  const [showRouteSelector, setShowRouteSelector] = useState(false);
  
  useEffect(() => {
    const loadRoutes = async () => {
      const routes = await fetchAllRoutes();
      setAvailableRoutes(routes || []);
    };
    loadRoutes();
  }, [fetchAllRoutes]);

  const filteredOrders = searchTerm 
    ? orders.filter(o => o.id.toLowerCase().includes(searchTerm.toLowerCase())) 
    : orders;
    
  const handleRouteSelect = (routeId) => {
    fetchRoute(routeId);
    setShowRouteSelector(false);
  };

  return (
    <section className="flex h-full flex-col bg-white">
      <header className="border-b border-neutral-200 px-4 py-4">
        <div className="flex flex-col">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-neutral-800">
              Ordenes
            </h2>
            <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-800">
              {orders.length} ordenes totales
            </span>
          </div>
          <div className="flex gap-2 mt-2">
            <button 
              onClick={() => {}}
              className="flex items-center justify-center py-1.5 px-3 text-xs font-medium rounded-md bg-gradient-to-r from-green-500 to-blue-500 text-white hover:bg-blue-700"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Nueva Orden
            </button>
            <button 
              onClick={() => setShowRouteSelector(!showRouteSelector)}
              className="flex items-center justify-center py-1.5 px-3 text-xs font-medium rounded-md bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100"
            >
              <Map className="h-3.5 w-3.5 mr-1" />
              Seleccionar Ruta
            </button>
            {optimized && (
              <div className="text-xs py-1.5 px-3 bg-emerald-100 text-emerald-800 rounded-md flex items-center">
                Distancia: {optimized.total_length} km
              </div>
            )}
          </div>
          
          {showRouteSelector && (
            <div className="mt-2 bg-blue-50 border border-blue-200 rounded-md p-3">
              <h3 className="text-xs font-medium text-blue-800 mb-2">Rutas disponibles:</h3>
              <div className="max-h-40 overflow-y-auto">
                {availableRoutes.length > 0 ? (
                  <div className="space-y-1">
                    {availableRoutes.map(route => (
                      <div 
                        key={route.route_id}
                        onClick={() => handleRouteSelect(route.route_id)}
                        className="text-xs bg-white p-2 rounded border border-blue-100 cursor-pointer hover:bg-blue-100 flex justify-between items-center"
                      >
                        <div>
                          <span className="font-medium">{route.depot.name}</span>
                          <span className="ml-2 text-gray-500">
                            ({route.orders_count} órdenes)
                          </span>
                        </div>
                        {route.is_optimized && (
                          <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full text-[10px]">
                            Optimizada
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-blue-700">No hay rutas disponibles</p>
                )}
              </div>
            </div>
          )}
          <div className="relative mt-3">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
              <Search className="h-4 w-4 text-neutral-400" />
            </div>
            <input
              type="search"
              className="block w-full rounded-md border border-neutral-300 bg-white py-2 pl-10 pr-3 text-sm placeholder-neutral-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              placeholder="Buscar Ordenes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
      </header>
      <div className="flex-1 space-y-0 overflow-y-auto">
        {filteredOrders.length > 0 ? (
          filteredOrders.map((o) => <OrdenCard key={o.id} order={o} />)
        ) : (
          <div className="flex h-40 items-center justify-center">
            <p className="text-sm text-neutral-500">No hay ordenes disponibles</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default OrderList;

