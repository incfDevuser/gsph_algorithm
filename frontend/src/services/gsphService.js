const API_URL = "http://127.0.0.1:8000";

/**
 * 
 * @param {Object} data - Datos para la optimización
 * @param {Object} data.depot - Punto de inicio (depósito)
 * @param {Array} data.orders - Lista de órdenes para entregar
 * @param {number} [data.time_budget_s=1.5] - Presupuesto de tiempo en segundos
 * @param {number} [data.seed=42] - Semilla para la generación aleatoria
 * @returns {Promise<Object>} Resultado de la optimización
 */
export const optimizeRoute = async (data) => {
  try {
    const response = await fetch(`${API_URL}/optimize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        depot: data.depot,
        orders: data.orders,
        time_budget_s: data.time_budget_s || 1.5,
        seed: data.seed || 42,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error al optimizar ruta: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error en la llamada a la API de GSPH:", error);
    throw error;
  }
};

/**
 * 
 * @returns {Promise<boolean>}
 */
export const checkApiStatus = async () => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    
    const response = await fetch(`${API_URL}`, { 
      signal: controller.signal 
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.warn("API de GSPH no disponible:", error);
    return false;
  }
};
