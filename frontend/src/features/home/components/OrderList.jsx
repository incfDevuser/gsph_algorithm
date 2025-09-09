import React, { useState } from "react";
import OrdenCard from "./OrdenCard";
import tspData from "../../../data/tsp_santiago.json";
import { Search, FileDown, Filter } from "lucide-react";

const OrderList = () => {
  const [searchTerm, setSearchTerm] = useState("");
  
  const filteredOrders = searchTerm 
    ? tspData.orders.filter(o => o.id.toLowerCase().includes(searchTerm.toLowerCase())) 
    : tspData.orders;

  return (
    <section className="flex h-full flex-col bg-white">
      <header className="border-b border-neutral-200 px-4 py-4">
        <div className="flex flex-col">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-neutral-800">
              Ordenes
            </h2>
            <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-800">
              {tspData.orders.length} ordenes totales
            </span>
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
            <p className="text-sm text-neutral-500">No orders match your search</p>
          </div>
        )}
      </div>
    </section>
  );
};

export default OrderList;

