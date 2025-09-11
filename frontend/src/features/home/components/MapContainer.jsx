import React, { useState } from "react";
import {
  MapContainer as LeafletMap,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MapStyles.css";
import CuadranteLayer from "./CuadranteLayer";
import RutaConectora from "./RutaConectora";
import { useGSPHStore } from "../../../store/GSPHStore";

const depotIcon = new L.DivIcon({
  className: "custom-div-icon",
  html: `<div style="background-color: #1e3a8a; width: 18px; height: 18px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"></div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

const createDeliveryIcon = (id) => {
  let color = "#ef4444";
  const lastChar = id.slice(-1);
  if ("456".includes(lastChar)) color = "#eab308";
  if ("789".includes(lastChar)) color = "#3b82f6";

  return new L.DivIcon({
    className: "custom-div-icon",
    html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
};

function MapClickHandler() {
  const { selectLocationMode, setSelectedLocation } = useGSPHStore();
  
  const map = useMapEvents({
    click: (e) => {
      if (selectLocationMode) {
        setSelectedLocation({
          lat: e.latlng.lat,
          lng: e.latlng.lng
        });
        console.log("Ubicación seleccionada:", e.latlng);
      } else {
        console.log("Clic en el mapa:", e.latlng);
      }
    }
  });
  
  return null;
}

const LocationMarker = () => {
  const { selectedLocation } = useGSPHStore();
  
  if (!selectedLocation) return null;
  
  return (
    <Marker 
      position={[selectedLocation.lat, selectedLocation.lng]}
      icon={new L.DivIcon({
        className: "custom-div-icon",
        html: `<div style="background-color: #10b981; width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9],
      })}
    >
      <Popup className="custom-popup">
        <div className="font-medium text-gray-900">Ubicación seleccionada</div>
        <div className="text-xs text-gray-600">
          {selectedLocation.lat.toFixed(6)}, {selectedLocation.lng.toFixed(6)}
        </div>
      </Popup>
    </Marker>
  );
};

const MapContainer = () => {
  const { route, optimized, selectedLocation } = useGSPHStore();
  const [showQuadrants, setShowQuadrants] = useState(false);
  
  const depot = route?.depot || { lat: -33.447487, lng: -70.673676, name: "Depot" };
  const orders = route?.orders || [];
  const center = [depot.lat, depot.lng];
  
  React.useEffect(() => {
    if (optimized) {
      setShowQuadrants(true);
    }
  }, [optimized]);

  const getOrderStatus = (id) => {
    const lastChar = id.slice(-1);
    if ("123".includes(lastChar)) return "Pendiente";
    if ("456".includes(lastChar)) return "Retirado";
    if ("789".includes(lastChar)) return "Programado";
    return "Pendiente";
  };

  return (
      <LeafletMap
        center={center}
        zoom={14}
        className="h-[60vh] md:h-[72vh] w-full"
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        <MapClickHandler />
        <Marker position={[depot.lat, depot.lng]} icon={depotIcon}>
          <Popup className="custom-popup">
            <div className="font-medium text-gray-900 mb-1">
              {depot.name || "Centro de Distribución"}
            </div>
            <div className="text-xs text-gray-600">
              Centro de operaciones principal
            </div>
          </Popup>
        </Marker>
        <CircleMarker
          center={[depot.lat, depot.lng]}
          radius={50}
          pathOptions={{
            fillColor: "#1e3a8a",
            fillOpacity: 0.05,
            weight: 1,
            color: "#1e3a8a",
            opacity: 0.3,
          }}
        />
        {orders.map((o) => {
          const status = getOrderStatus(o.id);
          return (
            <Marker
              key={o.id}
              position={[o.lat, o.lng]}
              icon={createDeliveryIcon(o.id)}
            >
            <Popup className="custom-popup">
              <div className="font-medium text-gray-900">
                Pedido #{o.id.substring(0, 5)}
              </div>
              <div className="text-xs text-gray-500 mb-1">Estado: {status}</div>
              <div className="text-xs text-gray-600">
                Coordenadas: {o.lat.toFixed(4)}, {o.lng.toFixed(4)}
              </div>
            </Popup>
          </Marker>
        );
      })}
      <CuadranteLayer visible={showQuadrants} />
      <RutaConectora visible={!!optimized} coordinates={optimized?.optimized_coords?.map(p => [p.lat, p.lng])} />
      <LocationMarker />
      <div className="leaflet-top leaflet-left" style={{ marginTop: "10px" }}>
        <div className="leaflet-control leaflet-bar bg-white p-2 rounded shadow-md">
          <div className="flex flex-col gap-2">
            <button
              onClick={() => setShowQuadrants(!showQuadrants)}
              className={`px-3 py-2 text-xs font-medium rounded ${
                showQuadrants
                  ? "bg-blue-100 text-blue-700 border border-blue-300"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              {showQuadrants
                ? "Ocultar Cuadrantes"
                : "Mostrar Cuadrantes"}
            </button>

            {optimized && (
              <div className="mt-1 px-3 py-1 text-xs font-medium bg-emerald-100 text-emerald-700 border border-emerald-300 rounded flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                Ruta GSPH activa
              </div>
            )}
          </div>
        </div>
      </div>
    </LeafletMap>
  );
};

export default MapContainer;
