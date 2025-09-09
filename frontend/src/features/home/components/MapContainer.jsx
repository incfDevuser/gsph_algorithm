import React from "react";
import { MapContainer as LeafletMap, TileLayer, Marker, Popup, CircleMarker } from "react-leaflet";
import L from "leaflet";
import data from "../../../data/tsp_santiago.json";
import "leaflet/dist/leaflet.css";
import "./MapStyles.css";

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

const MapContainer = () => {
  const center = [data.depot.lat, data.depot.lng];
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
      <Marker position={[data.depot.lat, data.depot.lng]} icon={depotIcon}>
        <Popup className="custom-popup">
          <div className="font-medium text-gray-900 mb-1">{data.depot.name || "Centro de Distribución"}</div>
          <div className="text-xs text-gray-600">Centro de operaciones principal</div>
        </Popup>
      </Marker>
      <CircleMarker 
        center={[data.depot.lat, data.depot.lng]}
        radius={50}
        pathOptions={{ fillColor: '#1e3a8a', fillOpacity: 0.05, weight: 1, color: '#1e3a8a', opacity: 0.3 }}
      />
      {data.orders.map((o) => {
        const status = getOrderStatus(o.id);
        return (
          <Marker key={o.id} position={[o.lat, o.lng]} icon={createDeliveryIcon(o.id)}>
            <Popup className="custom-popup">
              <div className="font-medium text-gray-900">Pedido #{o.id.substring(0, 5)}</div>
              <div className="text-xs text-gray-500 mb-1">Estado: {status}</div>
              <div className="text-xs text-gray-600">
                Coordenadas: {o.lat.toFixed(4)}, {o.lng.toFixed(4)}
              </div>
            </Popup>
          </Marker>
        );
      })}
    </LeafletMap>
  );
};

export default MapContainer;
