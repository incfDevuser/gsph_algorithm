import React from "react";
import { Polyline } from "react-leaflet";

const RutaConectoraNew = ({ visible = false, coordinates = [] }) => {
  if (!visible || !coordinates || coordinates.length < 2) return null;

  return (
    <Polyline
      positions={coordinates}
      pathOptions={{
        color: "#3b82f6", 
        weight: 4,
        opacity: 0.8,
        lineCap: "round",
        lineJoin: "round"
      }}
    />
  );
};

export default RutaConectoraNew;