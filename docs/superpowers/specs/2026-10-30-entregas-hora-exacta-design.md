# Entregas con hora exacta para varios repartidores — diseño

Fecha: 2026-10-30. Estado: diseño acordado en conversación, pendiente de plan.

## Intención y por qué

Hoy los delivery prometen rangos enormes ("entre 9 y 18", "entre 9 y 12") porque no confían en sus
propios tiempos: los errores de cada tramo y de cada entrega se acumulan a lo largo de la ruta. La meta
es un sistema que, para una flota de varios repartidores, entregue a cada uno su ruta y a cada cliente
una **hora que se cumple**: si se dice una hora, se llega a esa hora (±5 minutos); si se da un rango,
es corto (1 hora), nunca "de 9 a 6".

Para qué (en palabras del usuario):
1. Hacer **más entregas efectivas en el día**.
2. **Mejorar los tiempos de entrega** y con eso el contento de la gente.

Los dos objetivos chocan en un punto (promesas rígidas quitan flexibilidad a la ruta y obligan a
esperar) y se apoyan en otro (con hora exacta el cliente está en casa: menos entregas fallidas y
menos visitas repetidas). El sistema debe mostrar ese equilibrio con números.

Es una implementación pensada para el mundo real, **validada siempre en simulación** (no hay datos de
una flota real). Sus resultados alimentan el paper como extensión de GSPH a rutas urbanas reales.

## Lo que decidió el usuario

- **Varios repartidores**; cada uno recibe su ruta (orden de paradas) y la hora de cada entrega.
- **El sistema elige la hora** siempre. Modo opcional: el cliente elige entre las horas que el sistema
  *sí puede cumplir* (nunca entre horas que el sistema no podría sostener).
- **Tolerancia de la hora exacta: ±5 minutos.**
- **Rangos cortos** (del orden de 1 hora) o una hora específica; nunca rangos de medio día.
- **Siempre simulando**: no hay datos reales de repartos.
- **Lectura de negocio**: el resultado debe poder decirse como "si sumas un repartidor más, entregas X
  más por día". Por eso, en cada corrida la flota es **fija** y el turno **cierra a las 18:00** (sin horas
  extra): lo que no alcanza queda para el día siguiente. La evaluación repite el escenario variando la
  cantidad de repartidores (ver "Evaluación").
- Aprobó el esquema de cuatro piezas descrito abajo (mundo simulado, repartir y rutear, simular la
  ruta para prometer rangos, ejecución con hora exacta) y la evaluación por políticas.

## Metas

1. Planificar el día de una flota: repartir paradas entre K repartidores y rutear cada uno sobre la
   red vial real (OSRM), con tiempos dependientes de la hora.
2. Prometer a cada cliente un **rango de 1 hora** que se cumpla con alta probabilidad.
3. En el día, comunicar una **hora exacta** con anticipación y cumplirla con ±5 minutos.
4. Modo "el cliente elige": ofrecer solo los rangos factibles sin romper promesas ya hechas.
5. Medir en simulación, comparando políticas, cuánta precisión se gana y cuántas entregas por día
   cuesta (o ahorra).

## No-metas (por ahora)

- Datos reales de una empresa, tráfico real en vivo o APIs pagadas (Google, HERE, TomTom).
- App para repartidores o clientes, notificaciones reales, backend en producción.
- Capacidad de los vehículos (peso/volumen), múltiples depósitos, pausas legales del conductor.
- Pedidos que llegan **durante** el día de reparto con los repartidores ya en ruta (ver "Fases").

## Anti-metas (sería un fracaso aunque "funcione")

- **Prometer horas que no se cumplen.** Una hora exacta incumplida es peor que un rango honesto. Si el
  sistema no puede sostener una promesa, debe dar un rango más ancho o avisar, no fingir precisión.
- **Que el planificador vea la realidad.** En la simulación, el planificador solo conoce promedios
  (perfil de tráfico medio, tiempo de entrega típico); los tiempos realizados los genera el mundo
  simulado y nunca se filtran a la planificación. Si se filtran, los resultados no valen nada.
- **Precisión a costa de colapsar la eficiencia**: repartidores esperando gran parte del día para
  cumplir promesas. La espera debe medirse y reportarse siempre.
- Resultados no reproducibles: toda corrida debe poder repetirse con una semilla.

## Restricciones

- Construye sobre lo que ya existe en `experiments/real/`: servidor OSRM local (`setup_osrm.sh`,
  puerto 5001, mapa de Chile, penalización por semáforo), `osrm_cliente.py` (matrices de tiempo y
  distancia, geometría) y `solver_ruta.py` (ATSP con búsqueda local orientada + ILS, validado contra
  Held-Karp y mejor que OR-Tools en Santiago). `GSPH_BASE.py` y `GSPH_V2.py` no se tocan.
- Python con numpy/numba, como el resto del repo; corre en un laptop (sin servicios externos de pago).
- Código, comentarios, salidas y documentación en **español**.
- Salidas bajo `experiments/results/real/` (git-ignored).

## Cómo funciona (acordado)

### 1. Mundo simulado (la "realidad")

- **Tiempo de viaje** de cada tramo = tiempo base de OSRM × factor de tráfico según la hora de salida
  × ruido aleatorio. El ruido tiene una parte común a todo el día (hay días peores) y una parte propia
  de cada tramo.
- **Tiempo de entrega** en cada parada: variable, típicamente 2–4 minutos, con cola larga (a veces
  10 o más: edificios, cliente que no baja).
- **Presencia del cliente**: la probabilidad de que el cliente no esté depende de qué tan preciso fue
  lo prometido (rango largo → más ausencias). Una ausencia es una entrega fallida.
- El planificador ve solo los valores esperados: el perfil de tráfico medio por hora y el tiempo de
  entrega típico.

### 2. Repartir y rutear (VRP) rescatando GSPH

- **Asignación inicial por zonas geográficas**: la idea de partición de GSPH (cuadrantes/sectores)
  reparte las paradas entre los K repartidores.
- **Ruta de cada repartidor** con el solver de calles reales (`solver_ruta.py`), extendido a tiempos
  dependientes de la hora y a rangos de entrega.
- **Mejora entre rutas**: mover y cambiar paradas entre repartidores para equilibrar la carga y bajar
  el total.

### 3. Simular cada ruta para prometer rangos

- Cada ruta planificada se "maneja" muchas veces en el simulador (del orden de 500) con tráfico y
  tiempos de entrega muestreados. De ahí sale, para cada parada, la distribución de su hora de llegada.
- Se elige para cada parada el **rango de 1 hora** que cubre su llegada con alta probabilidad
  (objetivo: 95%).
- **Si el repartidor llega antes del inicio del rango, espera.** Esa espera "resincroniza" la ruta y
  evita que la incertidumbre se acumule sin límite en las paradas tardías.
- **Modo "el cliente elige"**: al entrar un pedido nuevo, se prueba insertarlo en cada ruta y cada
  posición; solo se ofrecen los rangos donde cabe sin que ninguna promesa ya hecha baje de la
  probabilidad objetivo. Las promesas ya hechas pasan a ser restricciones duras (ventanas del TSPTW/
  VRPTW).

### 4. Ejecución en el día: la hora exacta

- Cada vez que el repartidor termina una entrega, se recalcula lo que le queda con la hora real y el
  estado del momento.
- Cuando una parada está a unos **45 minutos**, se fija y comunica la **hora exacta**; el repartidor la
  cumple con ±5 minutos, esperando si va adelantado. La hora exacta siempre cae dentro del rango ya
  prometido.
- Si aparece un atraso, se reordena lo que queda respetando los rangos (TSPTW sobre las paradas
  restantes). Si no hay forma de cumplir, **se avisa al cliente apenas se detecta**, no al llegar tarde.

### Evaluación: tres políticas en el mismo mundo simulado

| Política | Qué se promete |
|---|---|
| P0 "como hoy" | rango de todo el día (9 a 18) |
| P1 | rango de 1 hora |
| P2 | rango de 1 hora + hora exacta ~45 min antes |

Métricas por política (sobre muchos días simulados): % de llegadas dentro del rango; % dentro de ±5
min de la hora exacta; entregas efectivas por repartidor al día; entregas fallidas; km recorridos;
tiempo de espera de los repartidores; avisos de atraso; pendientes que pasan al día siguiente. El
resultado central es la curva **precisión de la promesa vs. entregas efectivas por día**.

**Barrido de flota (lectura de negocio)**: con una demanda diaria dada, se corre cada política con K, K+1,
K+2, ... repartidores (cada corrida con flota fija y cierre a las 18:00). De ahí salen, por política:
- entregas efectivas por día según la cantidad de repartidores, y el **valor marginal de un repartidor
  más** ("+1 repartidor = +X entregas al día");
- la cantidad mínima de repartidores para atender la demanda del día sin acumular pendientes.

Esto también permite expresar la precisión en esos términos: por ejemplo, cuántos repartidores extra
"cuesta" o "ahorra" pasar de P0 a P2.

El día de reparto parte con las pendientes del día anterior más la demanda nueva. Qué paradas quedan
afuera cuando no cabe todo lo decide el planificador (el problema pasa a elegir qué paradas atender, no
solo en qué orden), dando prioridad a las que ya vienen atrasadas.

## Fases (decisión mía, ajustable)

1. **Mundo simulado + flota sin promesas (P0)**: tráfico por hora, tiempos de entrega, VRP de K
   repartidores con tiempos dependientes de la hora. Base para comparar.
2. **Rangos de 1 hora (P1)**: simulación Monte Carlo de cada ruta, elección de rangos, espera si se
   llega antes, solver con ventanas.
3. **Hora exacta y re-planificación en el día (P2)**.
4. **Modo "el cliente elige"**: los clientes reservan uno a uno **antes** del día de reparto (por
   ejemplo, durante el día anterior) eligiendo entre rangos factibles; el día de reparto ya parte con
   todas las paradas conocidas. Pedidos nuevos con repartidores ya en ruta quedan para después.

## Decisiones mías (el usuario puede cambiarlas)

- Rango de 1 hora; hora exacta fijada ~45 min antes; cobertura objetivo 95%; ~500 simulaciones por ruta.
- El repartidor espera si llega antes del rango o antes de la hora exacta menos 5 min.
- Prioridad a las pendientes atrasadas cuando no cabe todo; una parada pendiente recibe su rango
  prometido para el día en que efectivamente se planifica.
- Escala de prueba: 3 a 15 repartidores, 100 a 600 paradas por día, turno de 9:00 a 18:00, un depósito,
  en las zonas de Santiago ya definidas en `ciudad.py`.
- Perfil de tráfico **sintético** (OSRM no tiene tráfico): factor por hora con puntas de mañana y tarde,
  igual para toda la ciudad en la primera versión, parametrizable. Los números absolutos son
  ilustrativos; lo que se compara son las políticas entre sí bajo el mismo mundo.
- Probabilidad de ausencia del cliente según el ancho de lo prometido: valores supuestos y
  parametrizables (por ejemplo, mayor con rango de 9 horas que con 1 hora o con hora exacta); se
  reportará la sensibilidad de los resultados a esos valores.
- Sin capacidad de vehículos ni pedidos en vivo en esta versión (ver no-metas y fases).
- **Qué sabe el planificador de la incertidumbre**: igual que una empresa real, la **estima de días
  pasados**. Se simula un conjunto de días "históricos" (separado de los días de evaluación) y de ahí
  el planificador estima el perfil de tráfico medio, la dispersión por tramo y la distribución del
  tiempo de entrega. Nunca usa los parámetros internos del mundo simulado ni los valores realizados
  del día evaluado. Además se reporta un caso de sensibilidad donde el mundo se aparta de lo estimado
  (día con tráfico peor, cola de entrega más pesada) para mostrar cuánto se degrada la cobertura.
- **Ausencias y llegadas tarde**: la probabilidad de ausencia depende del ancho de lo prometido **y**
  sube si se llega fuera del rango o de la hora exacta. Un intento fallido cuesta un tiempo corto fijo
  (tocar el timbre y esperar), no el tiempo de entrega completo.
- **Reintentos**: una entrega fallida vuelve como parada al día siguiente. La evaluación corre
  secuencias de varios días seguidos con el arrastre de pendientes, no días independientes.
- **Si el 95% no cabe en 1 hora**: primero se intenta cumplir re-planificando con esos rangos como
  ventanas (reordenar, mover la parada a otro repartidor). Solo si aun así no cabe, esa parada recibe
  un rango de hasta 2 horas, y esos casos se cuentan y reportan aparte. La cobertura objetivo es **por
  parada**; también se reporta el % de rutas con todas sus promesas cumplidas.
- **Hora exacta**: objetivo de 95% de llegadas dentro de ±5 min. Si se detecta un atraso, se avisa con
  una nueva hora; para las métricas, **una entrega avisada cuenta como promesa incumplida** (se reporta
  también cuántas se avisaron a tiempo). El re-planificador minimiza promesas incumplidas.
- **Modo "el cliente elige"**: rangos fijos de reloj (10:00–11:00, 11:00–12:00, ...). Cada cliente
  simulado tiene una hora preferida (con más demanda en la mañana y al final de la tarde) y elige el
  rango factible más cercano a ella; los clientes reservan en orden aleatorio. Se compara contra el modo
  "el sistema elige" para medir cuánto cuesta dejar elegir.

## Lo que queda al constructor

- Estructura de módulos y nombres dentro de `experiments/real/`.
- Distribuciones exactas (forma y parámetros) del ruido de tráfico y del tiempo de entrega, dentro de
  lo descrito.
- Cómo extender el solver a tiempos dependientes de la hora y ventanas (penalización por atraso,
  holguras, evaluación por ruta), siempre que se mantenga validado (comparación contra óptimo exacto
  en instancias chicas y contra OR-Tools con ventanas en las grandes).
- Algoritmo concreto de asignación por zonas y de los movimientos entre rutas.
- Formato de salidas, mapas y tablas para el paper.
