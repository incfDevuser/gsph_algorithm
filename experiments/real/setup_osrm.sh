#!/usr/bin/env bash
# Prepara y levanta un servidor OSRM local con el mapa de Chile (OpenStreetMap).
#
# Uso (desde cualquier carpeta):
#   bash experiments/real/setup_osrm.sh            # descarga si falta, procesa si falta, levanta servidor
#   SEMAFORO_S=20 bash experiments/real/setup_osrm.sh --rebuild   # reprocesa con otra penalización por semáforo
#
# El servidor queda en http://localhost:5001 (contenedor docker "osrm-chile").
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$RAIZ/experiments/data/osm"
IMG="ghcr.io/project-osrm/osrm-backend:v6.0.0"
PBF_URL="https://download.geofabrik.de/south-america/chile-latest.osm.pbf"
NOMBRE="chile-latest"
# Penalización por semáforo en segundos (el perfil car.lua de OSRM trae 2 s, muy optimista para Santiago).
SEMAFORO_S="${SEMAFORO_S:-15}"
# 5000 lo ocupa el receptor AirPlay de macOS.
PUERTO="${PUERTO:-5001}"

mkdir -p "$DATA"

if [[ ! -f "$DATA/$NOMBRE.osm.pbf" ]]; then
  echo "Descargando mapa de Chile desde Geofabrik (~350 MB)..."
  curl -L -o "$DATA/$NOMBRE.osm.pbf" "$PBF_URL"
fi

if [[ "${1:-}" == "--rebuild" ]]; then
  rm -f "$DATA"/$NOMBRE.osrm*
fi

if [[ ! -f "$DATA/$NOMBRE.osrm.mldgr" ]]; then
  # Perfil de auto con penalización por semáforo: OSRM la lee de profile.properties
  # (lib/obstacles.lua) y si falta usa 2 s; se agrega junto a u_turn_penalty.
  docker run --rm "$IMG" cat /opt/car.lua \
    | awk -v s="$SEMAFORO_S" '{print} /u_turn_penalty *=/ {print "      traffic_signal_penalty         = " s ","}' \
    > "$DATA/car_semaforo.lua"
  grep -q "traffic_signal_penalty" "$DATA/car_semaforo.lua" || { echo "No se pudo ajustar el perfil"; exit 1; }
  # car.lua hace require('lib/...'), así que el perfil tiene que quedar junto a /opt/lib.
  echo "Procesando red vial (extract/partition/customize, semáforo = ${SEMAFORO_S}s)..."
  docker run --rm -v "$DATA:/data" -v "$DATA/car_semaforo.lua:/opt/car_semaforo.lua:ro" "$IMG" \
    osrm-extract -p /opt/car_semaforo.lua "/data/$NOMBRE.osm.pbf"
  docker run --rm -v "$DATA:/data" "$IMG" osrm-partition "/data/$NOMBRE.osrm"
  docker run --rm -v "$DATA:/data" "$IMG" osrm-customize "/data/$NOMBRE.osrm"
fi

docker rm -f osrm-chile >/dev/null 2>&1 || true
docker run -d --name osrm-chile -p "$PUERTO:5000" -v "$DATA:/data" "$IMG" \
  osrm-routed --algorithm mld --max-table-size 10000 --max-viaroute-size 10000 \
  --max-nearest-size 100 "/data/$NOMBRE.osrm" >/dev/null
echo "OSRM levantado en http://localhost:$PUERTO (contenedor osrm-chile)"
