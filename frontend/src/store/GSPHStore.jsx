import { create } from "zustand";
import axios from "axios";

export const useGSPHStore = create((set, get) => ({
  loading: false,
  error: null,
  route: null,
  optimized: null,
  selectedLocation: null,
  selectLocationMode: false,
    //crear la ruta
  createRoute: async (depot) => {
    set({ loading: true, error: null });
    try {
      const res = await axios.post("http://localhost:8000/routes/", {
        depot,
        orders: [],
      });
      set({ route: res.data, optimized: null });
    } catch (err) {
      set({ error: err.message || "Error al crear ruta" });
    } finally {
      set({ loading: false });
    }
  },
  //agregar ordenes a la ruta
  addOrders: async (orders) => {
    const route = get().route;
    if (!route) {
      set({ error: "No hay ruta cargada" });
      return;
    }
    set({ loading: true, error: null });
    try {
      const response = await axios.post(
        `http://localhost:8000/routes/${route.route_id}/orders`,
        orders
      );
      const updatedRoute = {
        ...route,
        orders: [...(route.orders || []), ...orders]
      };
      set({ route: updatedRoute });
    } catch (err) {
      set({ error: err.message || "Error al agregar órdenes" });
    } finally {
      set({ loading: false });
    }
  },
  //obtener la ruta, no optimizada
  fetchRoute: async (routeId) => {
    set({ loading: true, error: null });
    try {
      const res = await axios.get(`http://localhost:8000/routes/${routeId}`);
      set({ route: res.data });
    } catch (err) {
      set({ error: err.message || "Error al obtener la ruta" });
    } finally {
      set({ loading: false });
    }
  },
  //optimizar la ruta
  optimizeRoute: async () => {
    const route = get().route;
    if (!route) {
      set({ error: "No hay ruta cargada" });
      return;
    }
    set({ loading: true, error: null });
    try {
      const res = await axios.post(
        `http://localhost:8000/routes/${route.route_id}/optimize`
      );
      set({
        optimized: {
          optimized_coords: res.data.optimized_coords,
          total_length: res.data.total_length,
        },
      });
    } catch (err) {
      set({ error: err.message || "Error al optimizar la ruta" });
    } finally {
      set({ loading: false });
    }
  },
  //obtener la ruta optimizada
  fetchOptimized: async () => {
    const route = get().route;
    if (!route) {
      set({ error: "No hay ruta cargada" });
      return;
    }
    set({ loading: true, error: null });
    try {
      const res = await axios.get(
        `http://localhost:8000/routes/${route.route_id}/optimized`
      );
      set({ optimized: res.data });
    } catch (err) {
      set({ error: err.message || "Error al obtener ruta optimizada" });
    } finally {
      set({ loading: false });
    }
  },
  // Manejo de selección de ubicaciones
  setSelectedLocation: (location) => {
    set({ selectedLocation: location });
  },
  setSelectLocationMode: (mode) => {
    set({ selectLocationMode: mode });
  },
}));
