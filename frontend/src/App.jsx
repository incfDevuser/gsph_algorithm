import React, { useState, createContext, useEffect } from "react";
import Header from "./ui/Header";
import Map from "./features/home/components/Map";
import OrderList from "./features/home/components/OrderList";
import OrderForm from "./features/home/components/OrderForm";
import Modal from "./ui/Modal";
import { optimizeRoute, checkApiStatus } from "./services/gsphService";
import tspData from "./data/tsp_santiago.json";

export const GSPHContext = createContext();

function App() {
  const [gsphActive, setGsphActive] = useState(false);
  const [calculating, setCalculating] = useState(false);
  const [apiAvailable, setApiAvailable] = useState(true);
  const [orders, setOrders] = useState(tspData.orders);
  const [depot, setDepot] = useState(tspData.depot);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [showOrderForm, setShowOrderForm] = useState(false);

  useEffect(() => {
    const checkApi = async () => {
      const isAvailable = await checkApiStatus();
      setApiAvailable(isAvailable);
    };
    checkApi();
  }, []);

  const handleOrderCreated = (newOrder) => {
    setOrders([...orders, newOrder]);
    if (gsphActive) {
      setGsphActive(false);
      setOptimizationResult(null);
    }
  };

  const generateOptimalRoute = async () => {
    setCalculating(true);
    try {
      const result = await optimizeRoute({
        depot: depot,
        orders: orders,
        time_budget_s: 1.5
      });
      
      setOptimizationResult(result);
      setGsphActive(true);
    } catch (error) {
      console.error("Error al optimizar ruta:", error);
      alert("Error al optimizar la ruta. Verifica que la API esté funcionando correctamente.");
    } finally {
      setCalculating(false);
    }
  };

  return (
    <GSPHContext.Provider value={{ 
      gsphActive, 
      setGsphActive, 
      orders, 
      setOrders, 
      depot, 
      setDepot, 
      optimizationResult, 
      setOptimizationResult,
      handleOrderCreated,
      showOrderForm,
      setShowOrderForm
    }}>
      <div className="min-h-screen bg-neutral-50">
        <Header />
        <main className="mx-auto pt-26 max-w-7xl px-6 pb-8">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold text-neutral-800">
              Optimizador de Rutas para Entregas
            </h1>
            <p className="text-neutral-500 mt-1">
              Gestiona y optimiza las rutas de entrega con algoritmo GSPH
            </p>
            {!apiAvailable && (
              <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
                <p className="text-sm text-amber-700">
                  <strong>Advertencia:</strong> No se pudo conectar con la API de GSPH. Algunas funcionalidades pueden no estar disponibles.
                </p>
              </div>
            )}
          </div>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
            <section className="md:col-span-8 rounded-lg border border-neutral-200 bg-white shadow-sm overflow-hidden">
              <div className="p-4 border-b border-neutral-200 bg-white">
                <div className="flex justify-between items-center">
                  <h2 className="text-sm font-medium text-neutral-800">
                    Mapa de Entregas
                  </h2>
                  <div className="flex gap-2">
                    <button
                      onClick={generateOptimalRoute}
                      disabled={calculating}
                      className={`py-1.5 px-3 text-xs font-medium rounded-md ${
                        gsphActive
                          ? "bg-emerald-600 hover:bg-emerald-700"
                          : "bg-gradient-to-r from-green-500 to-blue-500 hover:opacity-90"
                      } text-white flex items-center`}
                    >
                      {calculating ? (
                        <>
                          <svg
                            className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Calculando...
                        </>
                      ) : gsphActive ? (
                        "Recalcular Ruta"
                      ) : (
                        "Generar Ruta Óptima"
                      )}
                    </button>
                  </div>
                </div>
              </div>
              <Map />
            </section>
            <aside className="md:col-span-4 h-full">
              <div className="max-h-[80vh] overflow-y-auto rounded-lg border border-neutral-200 shadow-sm bg-white">
                <OrderList />
              </div>
            </aside>
          </div>
        </main>
      </div>
      {showOrderForm && (
        <Modal>
          <OrderForm 
            onClose={() => setShowOrderForm(false)}
            onOrderCreated={handleOrderCreated} 
          />
        </Modal>
      )}
    </GSPHContext.Provider>
  );
}

export default App;
