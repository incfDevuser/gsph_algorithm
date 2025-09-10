import React from "react";
import { Rectangle, Tooltip } from "react-leaflet";
import data from "../../../data/tsp_santiago.json";

const CuadranteLayer = ({ visible = true }) => {
  if (!visible) return null;
  const allPoints = [
    [data.depot.lat, data.depot.lng],
    ...data.orders.map((o) => [o.lat, o.lng]),
  ];

  const lats = allPoints.map((p) => p[0]);
  const lngs = allPoints.map((p) => p[1]);

  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);

  const midLat = (minLat + maxLat) / 2;
  const midLng = (minLng + maxLng) / 2;
  const opacity = 0.15;
  const weight = 1;

  return (
    <>
      <Rectangle
        bounds={[
          [midLat, minLng],
          [maxLat, midLng],
        ]}
        pathOptions={{ color: "blue", fillOpacity: opacity, weight }}
      >
        <Tooltip direction="center" permanent>
          Q1 (NW)
        </Tooltip>
      </Rectangle>
      <Rectangle
        bounds={[
          [midLat, midLng],
          [maxLat, maxLng],
        ]}
        pathOptions={{ color: "green", fillOpacity: opacity, weight }}
      >
        <Tooltip direction="center" permanent>
          Q2 (NE)
        </Tooltip>
      </Rectangle>
      <Rectangle
        bounds={[
          [minLat, minLng],
          [midLat, midLng],
        ]}
        pathOptions={{ color: "red", fillOpacity: opacity, weight }}
      >
        <Tooltip direction="center" permanent>
          Q3 (SW)
        </Tooltip>
      </Rectangle>
      <Rectangle
        bounds={[
          [minLat, midLng],
          [midLat, maxLng],
        ]}
        pathOptions={{ color: "orange", fillOpacity: opacity, weight }}
      >
        <Tooltip direction="center" permanent>
          Q4 (SE)
        </Tooltip>
      </Rectangle>

      <Rectangle
        bounds={[
          [minLat, midLng - 0.0005],
          [maxLat, midLng + 0.0005],
        ]}
        pathOptions={{ color: "black", fillOpacity: 0.3, weight: 1 }}
      />
      <Rectangle
        bounds={[
          [midLat - 0.0005, minLng],
          [midLat + 0.0005, maxLng],
        ]}
        pathOptions={{ color: "black", fillOpacity: 0.3, weight: 1 }}
      />
    </>
  );
};

export default CuadranteLayer;
