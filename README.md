# Sistema de Optimización de Rutas con GSPH

Este proyecto es un sistema de optimización de rutas para entregas utilizando el algoritmo GSPH (Geometric Sweep Pulse Harmonization).

## Estructura del Proyecto

El proyecto consta de dos partes principales:

- **Frontend**: Aplicación web desarrollada con React y Tailwind CSS
- **Backend**: API REST desarrollada con FastAPI que implementa el algoritmo GSPH

## Requisitos

- Python 3.8 o superior
- Node.js 16 o superior
- npm o yarn

## Configuración

### Backend

1. Navega a la carpeta del backend:

```bash
cd backend
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta el servidor:

```bash
python main.py
```

El backend estará disponible en `http://127.0.0.1:8000`.

### Frontend

1. Navega a la carpeta del frontend:

```bash
cd frontend
```

2. Instala las dependencias:

```bash
npm install
# o
yarn install
```

3. Ejecuta el servidor de desarrollo:

```bash
npm run dev
# o
yarn dev
```

El frontend estará disponible en `http://localhost:3000`.

## Uso

1. Abre la aplicación web en tu navegador
2. Agrega órdenes de entrega utilizando el botón "Nueva Orden"
3. Configura la ubicación de cada orden en el mapa
4. Haz clic en "Generar Ruta Óptima" para calcular la ruta óptima usando GSPH
5. Visualiza la ruta en el mapa

## API

El backend expone un endpoint principal:

- `POST /optimize`: Recibe un conjunto de órdenes y devuelve la ruta óptima

### Ejemplo de solicitud:

```json
{
  "depot": {
    "id": "D-01",
    "name": "Local Central",
    "lat": -33.4489,
    "lng": -70.6693
  },
  "orders": [
    { "id": "O-01", "lat": -33.4569, "lng": -70.6483 },
    { "id": "O-02", "lat": -33.4429, "lng": -70.653 }
  ],
  "time_budget_s": 1.5,
  "seed": 42
}
```

### Ejemplo de respuesta:

```json
{
  "metrics": {
    "total_distance_km": 2.45,
    "nodes": 3
  },
  "solution": {
    "type": "FeatureCollection",
    "features": [...]
  }
}
```

## Características

- Visualización interactiva de rutas en mapa
- Cálculo de rutas óptimas con algoritmo GSPH
- Creación y gestión de órdenes de entrega
- Visualización de métricas de distancia total
