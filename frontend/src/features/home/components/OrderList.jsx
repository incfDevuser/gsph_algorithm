import React, { useState, useContext } from "react";
import OrdenCard from "./OrdenCard";
import { Search, FileDown, Filter, Plus } from "lucide-react";
import { GSPHContext } from "../../../App";

const OrderList = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const { orders, optimizationResult, setShowOrderForm } = useContext(GSPHContext);
  
  const filteredOrders = searchTerm 
    ? orders.filter(o => o.id.toLowerCase().includes(searchTerm.toLowerCase())) 
    : orders;

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
              onClick={() => setShowOrderForm(true)}
              className="flex items-center justify-center py-1.5 px-3 text-xs font-medium rounded-md bg-gradient-to-r from-green-500 to-blue-500 text-white hover:bg-blue-700"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Nueva Orden
            </button>
            {optimizationResult && (
              <div className="text-xs py-1.5 px-3 bg-emerald-100 text-emerald-800 rounded-md flex items-center">
                Distancia: {optimizationResult.metrics.total_distance_km} km
              </div>
            )}
          </div>
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

