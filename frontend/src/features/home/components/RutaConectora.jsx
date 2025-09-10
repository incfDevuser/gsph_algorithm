import React, { useContext } from "react";
import { Polyline, Marker, GeoJSON } from "react-leaflet";
import L from "leaflet";
import { GSPHContext } from "../../../App";

const simularGSPH = (points) => {
  const lats = points.map((p) => p[0]);
  const lngs = points.map((p) => p[1]);
  const midLat = (Math.min(...lats) + Math.max(...lats)) / 2;
  const midLng = (Math.min(...lngs) + Math.max(...lngs)) / 2;

  const q1 = points.filter((p) => p[0] > midLat && p[1] < midLng); // NW
  const q2 = points.filter((p) => p[0] > midLat && p[1] > midLng); // NE
  const q3 = points.filter((p) => p[0] < midLat && p[1] < midLng); // SW
  const q4 = points.filter((p) => p[0] < midLat && p[1] > midLng); // SE

  const rutas = {
    Q1: q1,
    Q2: q2,
    Q3: q3,
    Q4: q4,
  };
  const conexiones = [];
  const puntoMasCercano = (puntos, lat, lng) => {
    if (!puntos.length) return null;
    return puntos.reduce((cercano, punto) => {
      const distActual = Math.hypot(punto[0] - lat, punto[1] - lng);
      const distCercana = Math.hypot(cercano[0] - lat, cercano[1] - lng);
      return distActual < distCercana ? punto : cercano;
    }, puntos[0]);
  };

  if (q1.length && q2.length) {
    const p1 = puntoMasCercano(q1, midLat, midLng);
    const p2 = puntoMasCercano(q2, midLat, midLng);
    conexiones.push([p1, p2]);
  }
  if (q1.length && q3.length) {
    const p1 = puntoMasCercano(q1, midLat, midLng);
    const p3 = puntoMasCercano(q3, midLat, midLng);
    conexiones.push([p1, p3]);
  }
  if (q2.length && q4.length) {
    const p2 = puntoMasCercano(q2, midLat, midLng);
    const p4 = puntoMasCercano(q4, midLat, midLng);
    conexiones.push([p2, p4]);
  }
  if (q3.length && q4.length) {
    const p3 = puntoMasCercano(q3, midLat, midLng);
    const p4 = puntoMasCercano(q4, midLat, midLng);
    conexiones.push([p3, p4]);
  }

  return { rutas, conexiones, midLat, midLng };
};

const RutaConectora = ({ visible = true }) => {
  const { depot, orders, optimizationResult } = useContext(GSPHContext);

  if (!visible) return null;

  const conexionIcon = new L.DivIcon({
    className: "custom-div-icon",
    html: `<div style="background-color: purple; width: 10px; height: 10px; border-radius: 50%; border: 2px solid white;"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });

  if (optimizationResult && optimizationResult.solution) {
    return (
      <GeoJSON
        data={optimizationResult.solution}
        style={(feature) => {
          switch (feature.properties.type) {
            case "route":
              return {
                color: "#10b981",
                weight: 4,
                opacity: 0.8,
              };
            case "connection":
              return {
                color: "purple",
                weight: 3,
                opacity: 0.8,
              };
            case "quadrant":
              return {
                color: "#3b82f6",
                fillColor: "#3b82f6",
                weight: 1,
                opacity: 0.4,
                fillOpacity: 0.1,
              };
            default:
              return {
                color: "#10b981",
                weight: 3,
              };
          }
        }}
      />
    );
  }
  const allPoints = [
    [depot.lat, depot.lng],
    ...orders.map((o) => [o.lat, o.lng]),
  ];

  const { rutas, conexiones } = simularGSPH(allPoints);
  const colors = {
    Q1: "blue",
    Q2: "green",
    Q3: "red",
    Q4: "orange",
  };

  return (
    <>
      {Object.entries(rutas).map(([quadName, points]) => {
        if (points.length < 2) return null;
        return (
          <Polyline
            key={quadName}
            positions={points}
            pathOptions={{
              color: colors[quadName],
              weight: 3,
              opacity: 0.8,
              dashArray: "5, 5",
            }}
          />
        );
      })}
      {conexiones.map((conn, idx) => (
        <React.Fragment key={`conn-${idx}`}>
          <Polyline
            positions={conn}
            pathOptions={{ color: "purple", weight: 3, opacity: 0.8 }}
          />
          <Marker position={conn[0]} icon={conexionIcon} />
          <Marker position={conn[1]} icon={conexionIcon} />
        </React.Fragment>
      ))}
    </>
  );
};

export default RutaConectora;
