import React, { useState } from "react";
import MapContainer from "./MapContainer";
import { MapPin, Layers, Navigation, Info, Grid, Route } from "lucide-react";

const Map = () => {
  const [activeLayer, setActiveLayer] = useState("default");

  const handleLayerChange = (layer) => {
    setActiveLayer(layer);
  };

  return (
    <div className="relative w-full h-full">
      <MapContainer />

      <div className="absolute bottom-4 right-4 bg-white rounded-lg shadow-md p-1.5 z-[1000]">
        <div className="flex flex-col gap-1.5">
          <button
            className="flex items-center justify-center w-8 h-8 rounded hover:bg-neutral-100 text-neutral-600"
            title="Centrar mapa"
          >
            <Navigation className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Map;
