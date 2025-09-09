import Header from "./ui/Header";
import Map from "./features/home/components/Map";
import OrderList from "./features/home/components/OrderList";

function App() {
  return (
    <div className="min-h-screen bg-neutral-50">
      <Header />
      <main className="mx-auto pt-26 max-w-7xl px-6 pb-8">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-neutral-800">Optimizador de Rutas para Entregas</h1>
          <p className="text-neutral-500 mt-1">Gestiona y optimiza las rutas de entrega de manera eficiente</p>
        </div>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <section className="md:col-span-8 rounded-lg border border-neutral-200 bg-white shadow-sm overflow-hidden">
            <div className="p-4 border-b border-neutral-200 bg-white">
              <div className="flex justify-between items-center">
                <h2 className="text-sm font-medium text-neutral-800">Mapa de Entregas</h2>
                <div className="flex gap-2">
                  <button className="py-1.5 px-3 text-xs font-medium rounded-md bg-gradient-to-r from-green-500 to-blue-500 text-white hover:bg-emerald-700">
                    Generar Ruta Optima
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
  );
}

export default App;
