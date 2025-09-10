import React, { useState, useContext, useEffect } from "react";
import { X, MapPin, Plus, Clock } from "lucide-react";
import { MapContext } from "./MapContainer";

const OrderForm = ({ onClose, onOrderCreated }) => {
  const [formData, setFormData] = useState({
    id: `O-${String(Date.now()).substring(7)}`,
    name: "",
    address: "",
    timeWindow: "08:00-10:00",
    lat: null,
    lng: null,
  });
  const [step, setStep] = useState(1);
  const [localSelectedLocation, setLocalSelectedLocation] = useState(null);
  const [localSelectLocationMode, setLocalSelectLocationMode] = useState(false);
  const mapContext = useContext(MapContext);

  const selectLocationMode =
    mapContext?.selectLocationMode || localSelectLocationMode;
  const setSelectLocationMode =
    mapContext?.setSelectLocationMode || setLocalSelectLocationMode;
  const selectedLocation =
    mapContext?.selectedLocation || localSelectedLocation;
  const setSelectedLocation =
    mapContext?.setSelectedLocation || setLocalSelectedLocation;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleLocationSelect = () => {
    setSelectLocationMode(true);
    setStep(2);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    let lat = selectedLocation?.lat || formData.lat;
    let lng = selectedLocation?.lng || formData.lng;

    if (!lat || !lng) {
      lat = -33.45 + (Math.random() * 0.1 - 0.05);
      lng = -70.65 + (Math.random() * 0.1 - 0.05);
      console.warn(
        "Usando ubicación aleatoria ya que no se seleccionó una ubicación"
      );
    }

    const newOrder = {
      ...formData,
      lat,
      lng,
    };

    onOrderCreated(newOrder);
    onClose();
    if (setSelectedLocation) {
      setSelectedLocation(null);
    }
    if (setSelectLocationMode) {
      setSelectLocationMode(false);
    }
  };

  return (
    <div className="bg-black/50 absolute inset-0 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md" style={{ position: 'relative', zIndex: 10000 }}>
        <div className="flex items-center justify-between p-4">
          <h3 className="font-semibold text-lg text-neutral-800">
            {step === 1 ? "Nueva orden de entrega" : "Selecciona ubicación"}
          </h3>
          <button
            onClick={onClose}
            className="text-neutral-500 hover:text-neutral-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {step === 1 ? (
          <form onSubmit={handleLocationSelect} className="p-4 space-y-4">
            <div className="space-y-1">
              <label
                htmlFor="id"
                className="text-sm font-medium text-neutral-700"
              >
                ID de orden
              </label>
              <input
                id="id"
                name="id"
                type="text"
                value={formData.id}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="name"
                className="text-sm font-medium text-neutral-700"
              >
                Nombre del cliente
              </label>
              <input
                id="name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="address"
                className="text-sm font-medium text-neutral-700"
              >
                Dirección
              </label>
              <input
                id="address"
                name="address"
                type="text"
                value={formData.address}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="timeWindow"
                className="text-sm font-medium text-neutral-700"
              >
                Ventana horaria
              </label>
              <select
                id="timeWindow"
                name="timeWindow"
                value={formData.timeWindow}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="08:00-10:00">08:00 - 10:00</option>
                <option value="10:00-12:00">10:00 - 12:00</option>
                <option value="12:00-14:00">12:00 - 14:00</option>
                <option value="14:00-16:00">14:00 - 16:00</option>
                <option value="16:00-18:00">16:00 - 18:00</option>
              </select>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-md hover:bg-blue-700 transition"
              >
                <MapPin className="h-4 w-4" />
                Seleccionar ubicación
              </button>
            </div>
          </form>
        ) : (
          <div className="p-4 space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-md p-3 text-sm text-blue-800">
              <div className="flex items-start">
                <MapPin className="h-5 w-5 text-blue-500 mr-2 mt-0.5" />
                {mapContext ? (
                  <p>
                    Haz clic en el mapa para seleccionar la ubicación de
                    entrega.
                  </p>
                ) : (
                  <p>
                    El selector de mapa no está disponible. Puedes continuar y
                    se asignará una ubicación automáticamente.
                  </p>
                )}
              </div>
            </div>

            {selectedLocation && (
              <div className="border border-neutral-200 rounded-md p-3 bg-neutral-50">
                <p className="text-sm font-medium text-neutral-700 mb-1">
                  Ubicación seleccionada:
                </p>
                <p className="text-sm text-neutral-600">
                  Lat: {selectedLocation.lat.toFixed(6)}, Lng:{" "}
                  {selectedLocation.lng.toFixed(6)}
                </p>
              </div>
            )}

            <div className="flex items-center justify-between pt-2">
              <button
                onClick={() => {
                  setStep(1);
                  if (setSelectLocationMode) {
                    setSelectLocationMode(false);
                  }
                }}
                className="py-2 px-4 border border-neutral-300 text-neutral-700 rounded-md hover:bg-neutral-50"
              >
                Volver
              </button>

              <button
                onClick={handleSubmit}
                disabled={mapContext && !selectedLocation}
                className={`py-2 px-4 rounded-md ${
                  !mapContext || selectedLocation
                    ? "bg-green-600 hover:bg-green-700 text-white"
                    : "bg-neutral-300 text-neutral-500 cursor-not-allowed"
                }`}
              >
                Crear orden
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OrderForm;
