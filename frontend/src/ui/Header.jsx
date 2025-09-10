import React from "react";
import { MapPin, BarChart3, Bell, Settings } from "lucide-react";

const Header = () => {
  return (
    <div className="pointer-events-none fixed left-0 top-4 z-50 w-full px-4">
      <header className="pointer-events-auto rounded-2xl border-2 flex items-center justify-between bg-white py-3 px-6 shadow-md border-b border-neutral-200">
        <div className="flex items-center gap-2">
          <div className="flex items-center">
            <span className="ml-2 text-lg font-semibold tracking-tight text-gray-800">
              GSPH{" "}
              <span className="bg-gradient-to-r from-green-500 to-blue-500 bg-clip-text text-transparent">
                Optimizador de Delivery
              </span>
            </span>
          </div>
        </div>
        <nav className="hidden md:flex items-center gap-6">
          <button
            className="group inline-flex items-center gap-2 text-sm font-medium transition text-gray-800 hover:text-blue-700"
            aria-label="Vista principal"
          >
            <MapPin className="h-4 w-4" />
            <span>Dashboard</span>
          </button>
          <button
            className="group inline-flex items-center gap-2 text-sm font-medium transition text-gray-500 hover:text-blue-700"
            aria-label="Analíticas"
          >
            <BarChart3 className="h-4 w-4" />
            <span>Analitica</span>
          </button>
        </nav>
        <div className="flex items-center gap-4">
          <button
            className="rounded-full p-1.5 text-gray-500 transition hover:bg-neutral-100 hover:text-gray-700"
            aria-label="Notificaciones"
          >
            <Bell className="h-5 w-5" />
          </button>
          <button
            className="rounded-full p-1.5 text-gray-500 transition hover:bg-neutral-100 hover:text-gray-700"
            aria-label="Configuración"
          >
            <Settings className="h-5 w-5" />
          </button>
          <div
            className="ml-1 flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-r from-green-500 to-blue-500 text-sm font-medium text-white"
            aria-label="Usuario"
            title="Martín"
          >
            M
          </div>
        </div>
      </header>
    </div>
  );
};

export default Header;
