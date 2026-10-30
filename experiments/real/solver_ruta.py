"""Solver de rutas sobre red vial real: TSP asimétrico (ATSP) con matriz de tiempos.

Rescata de GSPH_V2 lo que funciona y lo adapta a costos asimétricos
(calles de un sentido, giros, semáforos => D[a, b] != D[b, a]):

- Listas de candidatos por cercanía real (min de ambos sentidos).
- Búsqueda local con cola de nodos activos (don't-look bits):
  * 2-opt orientado: cada par de aristas se prueba en sus dos reconexiones
    (invertir el tramo o su complemento, que en ATSP son tours distintos). El
    costo de los arcos internos invertidos se obtiene en O(1) con sumas
    prefijas hacia adelante (F) y hacia atrás (B).
  * Or-opt (segmentos de 1..3) insertado en ambas orientaciones.
- ILS con double-bridge local (no invierte tramos: ideal para ATSP) y
  aceptación con temperatura, como V2.
- Embudo multi-arranque: vecino más cercano, orden geométrico por cuadrantes
  recursivos (la idea de partición de GSPH llevada a curva de Hilbert) e
  inserciones aleatorias; todos reciben búsqueda local y el mejor recibe el ILS.

Pensado para rutas reales (decenas a pocos miles de paradas): cada movimiento
aplicado cuesta O(n) (inversión + recálculo de prefijos), a cambio de que la
evaluación de cada vecino sea O(1) y exacta.
"""
import time

import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        # Respaldo sin numba: el decorador no hace nada y los kernels corren en Python.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco

CAND_M        = 12     # Vecinos candidatos por nodo
OR_OPT_MAX_L  = 3      # Largo máximo de segmento en Or-opt
N_INSERCION   = 4      # Arranques por inserción aleatoria en el embudo
DB_WINDOW_MIN = 50     # Largo máximo de cada tramo del double-bridge: max(MIN, n / DIV)
DB_WINDOW_DIV = 8
ILS_BLOCK     = 500    # Kicks por llamada al kernel (entre chequeos de reloj)
ACCEPT_TEMP   = 0.2    # Temperatura de aceptación, en costos promedio de arco
STALL_MIN     = 5000   # Kicks sin mejora antes de parar: max(MIN, FACTOR * n)
STALL_FACTOR  = 300


# ---------------------------------------------------------------------------
# Kernels (numba). Tour = arreglo de nodos en orden de visita, pos[nodo] = índice.
# F[k] = costo de tour[0] -> ... -> tour[k] (con F[n] = largo del ciclo);
# B[k] = lo mismo recorriendo cada arco al revés.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _semilla(seed):
    np.random.seed(seed)

@njit(cache=True)
def _largo(tour, D):
    n = tour.shape[0]
    s = 0
    for t in range(n):
        s += D[tour[t], tour[(t + 1) % n]]
    return s

@njit(cache=True)
def _prefijos(tour, D, F, B):
    n = tour.shape[0]
    F[0] = 0
    B[0] = 0
    for t in range(n):
        u = tour[t]
        v = tour[(t + 1) % n]
        F[t + 1] = F[t] + D[u, v]
        B[t + 1] = B[t] + D[v, u]

@njit(cache=True)
def _tramo(P, i, j, n):
    """Costo del camino de la posición i a la j hacia adelante (con vuelta), según prefijo P."""
    if j >= i:
        return P[j] - P[i]
    return P[n] - P[i] + P[j]

@njit(cache=True)
def _invertir(tour, pos, i, j):
    """Invierte el tramo circular de posiciones i..j (siempre ese tramo, nunca el complemento)."""
    n = tour.shape[0]
    L = (j - i) % n + 1
    for k in range(L // 2):
        a = (i + k) % n
        b = (j - k) % n
        va = tour[a]
        vb = tour[b]
        tour[a] = vb
        pos[vb] = a
        tour[b] = va
        pos[va] = b

@njit(cache=True)
def _mover_segmento(tour, pos, tmp, segbuf, i, L, u, rev):
    """Saca el segmento de posiciones i..i+L-1 y lo reinserta justo después del nodo u."""
    n = tour.shape[0]
    for k in range(L):
        segbuf[k] = tour[(i + k) % n]
    q = 0
    t = (i + L) % n
    for _ in range(n - L):
        x = tour[t]
        tmp[q] = x
        q += 1
        if x == u:
            for k in range(L):
                tmp[q] = segbuf[L - 1 - k] if rev else segbuf[k]
                q += 1
        t = (t + 1) % n
    for k in range(n):
        tour[k] = tmp[k]
        pos[tmp[k]] = k

@njit(cache=True)
def _push(v, qbuf, inq, head, cnt):
    if inq[v] == 0:
        inq[v] = 1
        qbuf[(head + cnt) % qbuf.shape[0]] = v
        cnt += 1
    return cnt

@njit(cache=True)
def _busqueda_local(tour, pos, D, cand, ncand, init, ninit, qbuf, inq, F, B, tmp, segbuf, or_lmax):
    """2-opt orientado + Or-opt sobre los nodos de 'init' y los que reactivan los movimientos.

    Supone F y B al día con 'tour'. Devuelve la ganancia total (el costo baja en eso).
    """
    n = tour.shape[0]
    head = 0
    cnt = 0
    for t in range(ninit):
        cnt = _push(init[t], qbuf, inq, head, cnt)
    total = 0
    while cnt > 0:
        a = qbuf[head]
        head = (head + 1) % n
        cnt -= 1
        inq[a] = 0

        pa = pos[a]
        sa = tour[(pa + 1) % n]
        pr = tour[(pa - 1) % n]
        d_as = D[a, sa]
        d_pa = D[pr, a]
        lim = max(d_as, d_pa)
        best = 0
        mtype = 0      # 1 = inversión de mi..mj, 2 = Or-opt
        mi = -1
        mj = -1
        mL = 0
        mu = -1
        mrev = False

        # 2-opt: pares de aristas {(a,sa),(c,sc)} y {(pr,a),(pc,c)}, cada uno en sus dos reconexiones.
        for k in range(ncand[a]):
            c = cand[a, k]
            dac = D[a, c]
            dca = D[c, a]
            if min(dac, dca) >= lim:
                break
            if c == sa or c == pr:
                continue
            pcp = pos[c]
            sc = tour[(pcp + 1) % n]
            pc = tour[(pcp - 1) % n]
            if dac < d_as:
                # a->c, sa->sc: invierte sa..c
                i = (pa + 1) % n
                g = d_as + D[c, sc] - dac - D[sa, sc] + _tramo(F, i, pcp, n) - _tramo(B, i, pcp, n)
                if g > best:
                    best, mtype, mi, mj = g, 1, i, pcp
            if dca < d_as:
                # c->a, sc->sa: invierte sc..a
                i = (pcp + 1) % n
                g = d_as + D[c, sc] - dca - D[sc, sa] + _tramo(F, i, pa, n) - _tramo(B, i, pa, n)
                if g > best:
                    best, mtype, mi, mj = g, 1, i, pa
            if dac < d_pa:
                # pr->pc, a->c: invierte a..pc
                j = (pcp - 1) % n
                g = d_pa + D[pc, c] - D[pr, pc] - dac + _tramo(F, pa, j, n) - _tramo(B, pa, j, n)
                if g > best:
                    best, mtype, mi, mj = g, 1, pa, j
            if dca < d_pa:
                # pc->pr, c->a: invierte c..pr
                j = (pa - 1) % n
                g = d_pa + D[pc, c] - D[pc, pr] - dca + _tramo(F, pcp, j, n) - _tramo(B, pcp, j, n)
                if g > best:
                    best, mtype, mi, mj = g, 1, pcp, j

        # Or-opt: segmentos de largo 1..or_lmax con a en un extremo, a quedar junto a un candidato.
        for L in range(1, or_lmax + 1):
            if L + 3 > n:
                break
            for side in range(2):
                if side == 1 and L == 1:
                    break
                i = pa if side == 0 else (pa - L + 1) % n
                e = (i + L - 1) % n
                s1 = tour[i]
                s2 = tour[e]
                p = tour[(i - 1) % n]
                nx = tour[(i + L) % n]
                g_rem = D[p, s1] + D[s2, nx] - D[p, nx]
                if g_rem <= best:
                    continue
                dinv = _tramo(B, i, e, n) - _tramo(F, i, e, n)   # extra por recorrer el segmento al revés
                for k in range(ncand[a]):
                    c = cand[a, k]
                    if min(D[a, c], D[c, a]) >= g_rem - best:
                        break
                    if (pos[c] - i) % n < L:
                        continue
                    # Entre c y su sucesor, con a pegado a c (arco c->a).
                    v = tour[(pos[c] + 1) % n]
                    if (pos[v] - i) % n >= L:
                        if a == s1:
                            g = g_rem - (D[c, s1] + D[s2, v] - D[c, v])
                            rv = False
                        else:
                            g = g_rem - (D[c, s2] + D[s1, v] - D[c, v] + dinv)
                            rv = True
                        if g > best:
                            best, mtype, mi, mL, mu, mrev = g, 2, i, L, c, rv
                    # Entre el predecesor de c y c, con a pegado a c (arco a->c).
                    u = tour[(pos[c] - 1) % n]
                    if (pos[u] - i) % n >= L:
                        if a == s2:
                            g = g_rem - (D[u, s1] + D[s2, c] - D[u, c])
                            rv = False
                        else:
                            g = g_rem - (D[u, s2] + D[s1, c] - D[u, c] + dinv)
                            rv = True
                        if g > best:
                            best, mtype, mi, mL, mu, mrev = g, 2, i, L, u, rv

        if mtype == 0:
            continue
        total += best
        if mtype == 1:
            x = tour[(mi - 1) % n]
            y = tour[(mj + 1) % n]
            m1 = tour[mi]
            m2 = tour[mj]
            _invertir(tour, pos, mi, mj)
            cnt = _push(x, qbuf, inq, head, cnt)
            cnt = _push(m1, qbuf, inq, head, cnt)
            cnt = _push(m2, qbuf, inq, head, cnt)
            cnt = _push(y, qbuf, inq, head, cnt)
        else:
            s1 = tour[mi]
            s2 = tour[(mi + mL - 1) % n]
            p = tour[(mi - 1) % n]
            nx = tour[(mi + mL) % n]
            v = tour[(pos[mu] + 1) % n]
            _mover_segmento(tour, pos, tmp, segbuf, mi, mL, mu, mrev)
            cnt = _push(p, qbuf, inq, head, cnt)
            cnt = _push(nx, qbuf, inq, head, cnt)
            cnt = _push(s1, qbuf, inq, head, cnt)
            cnt = _push(s2, qbuf, inq, head, cnt)
            cnt = _push(mu, qbuf, inq, head, cnt)
            cnt = _push(v, qbuf, inq, head, cnt)
        cnt = _push(a, qbuf, inq, head, cnt)
        _prefijos(tour, D, F, B)
    return total

@njit(cache=True)
def _ils_bloque(tour, pos, D, cand, ncand, cur, best_len, kicks, window, stall, temp,
                qbuf, inq, F, B, tmp, segbuf, or_lmax, best, saved, ends):
    """'kicks' iteraciones de ILS: double-bridge local + búsqueda local acotada.

    Acepta si el costo queda <= actual o, si empeora, con probabilidad exp(-delta/temp).
    Al rechazar, restaura la copia previa al kick. 'best' guarda el mejor tour visto.
    """
    n = tour.shape[0]
    for _ in range(kicks):
        for t in range(n):
            saved[t] = tour[t]
        L1 = np.random.randint(1, window + 1)
        L2 = np.random.randint(1, window + 1)
        L3 = np.random.randint(1, window + 1)
        e0 = np.random.randint(0, n)
        e1 = e0 + L1
        e2 = e1 + L2
        e3 = e2 + L3
        a0 = tour[e0 % n]
        b0 = tour[(e0 + 1) % n]
        a1 = tour[e1 % n]
        b1 = tour[(e1 + 1) % n]
        a2 = tour[e2 % n]
        b2 = tour[(e2 + 1) % n]
        a3 = tour[e3 % n]
        b3 = tour[(e3 + 1) % n]
        delta = (D[a0, b2] + D[a3, b1] + D[a2, b0] + D[a1, b3]) - \
                (D[a0, b0] + D[a1, b1] + D[a2, b2] + D[a3, b3])
        # A B C Dd E -> A Dd C B E (los tramos conservan su sentido).
        W = L1 + L2 + L3
        for t in range(W):
            tmp[t] = tour[(e0 + 1 + t) % n]
        q = e0 + 1
        for t in range(L1 + L2, W):
            v = tmp[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        for t in range(L1, L1 + L2):
            v = tmp[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        for t in range(L1):
            v = tmp[t]
            tour[q % n] = v
            pos[v] = q % n
            q += 1
        ends[0] = a0
        ends[1] = b0
        ends[2] = a1
        ends[3] = b1
        ends[4] = a2
        ends[5] = b2
        ends[6] = a3
        ends[7] = b3
        _prefijos(tour, D, F, B)
        gain = _busqueda_local(tour, pos, D, cand, ncand, ends, 8, qbuf, inq, F, B, tmp, segbuf, or_lmax)
        new_len = cur + delta - gain
        accept = new_len <= cur
        if not accept and temp > 0.0:
            accept = np.random.random() < np.exp(-(new_len - cur) / temp)
        if accept:
            cur = new_len
            if new_len < best_len:
                best_len = new_len
                stall = 0
                for t in range(n):
                    best[t] = tour[t]
            else:
                stall += 1
        else:
            for t in range(n):
                v = saved[t]
                tour[t] = v
                pos[v] = t
            _prefijos(tour, D, F, B)
            stall += 1
    return cur, best_len, stall

@njit(cache=True)
def _vecino_mas_cercano(D, inicio):
    n = D.shape[0]
    tour = np.empty(n, np.int64)
    usado = np.zeros(n, np.bool_)
    tour[0] = inicio
    usado[inicio] = True
    for k in range(1, n):
        u = tour[k - 1]
        mejor = -1
        md = 0
        for v in range(n):
            if not usado[v] and (mejor < 0 or D[u, v] < md):
                mejor = v
                md = D[u, v]
        tour[k] = mejor
        usado[mejor] = True
    return tour

@njit(cache=True)
def _insercion(D, orden):
    """Inserta los nodos en el orden dado, cada uno en su arco más barato (lista enlazada)."""
    n = orden.shape[0]
    nxt = np.full(D.shape[0], -1, np.int64)
    a = orden[0]
    b = orden[1]
    nxt[a] = b
    nxt[b] = a
    for k in range(2, n):
        x = orden[k]
        mejor_u = a
        mejor = D[a, x] + D[x, nxt[a]] - D[a, nxt[a]]
        u = nxt[a]
        while u != a:
            c = D[u, x] + D[x, nxt[u]] - D[u, nxt[u]]
            if c < mejor:
                mejor = c
                mejor_u = u
            u = nxt[u]
        nxt[x] = nxt[mejor_u]
        nxt[mejor_u] = x
    tour = np.empty(n, np.int64)
    u = a
    for k in range(n):
        tour[k] = u
        u = nxt[u]
    return tour


# ---------------------------------------------------------------------------
# Envoltorio en Python
# ---------------------------------------------------------------------------

def _orden_hilbert(xy, bits=16):
    """Orden de visita por curva de Hilbert: cuadrantes recursivos en el orden Q1-Q2-Q4-Q3 de GSPH."""
    xy = np.asarray(xy, float)
    lo = xy.min(axis=0)
    span = max(float((xy.max(axis=0) - lo).max()), 1e-12)
    m = (1 << bits) - 1
    g = np.floor((xy - lo) / span * m).astype(np.int64)
    claves = np.empty(len(g), np.int64)
    for idx, (x, y) in enumerate(g):
        d = 0
        s = 1 << (bits - 1)
        while s > 0:
            rx = 1 if (x & s) else 0
            ry = 1 if (y & s) else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x = m - x
                    y = m - y
                x, y = y, x
            s >>= 1
        claves[idx] = d
    return np.argsort(claves, kind='stable').astype(np.int64)


def candidatos(D, m=CAND_M):
    """Los m nodos más cercanos a cada uno según min(D[a,c], D[c,a]), ordenados de menor a mayor."""
    n = D.shape[0]
    m = min(m, n - 1)
    S = np.minimum(D, D.T).astype(np.float64)
    np.fill_diagonal(S, np.inf)
    idx = np.argpartition(S, m - 1, axis=1)[:, :m]
    orden = np.take_along_axis(S, idx, axis=1).argsort(axis=1, kind='stable')
    cand = np.ascontiguousarray(np.take_along_axis(idx, orden, axis=1).astype(np.int64))
    return cand, np.full(n, m, np.int64)


class _Espacio:
    """Buffers reutilizables para los kernels."""
    def __init__(self, n):
        self.qbuf = np.zeros(n, np.int64)
        self.inq = np.zeros(n, np.int64)
        self.F = np.zeros(n + 1, np.int64)
        self.B = np.zeros(n + 1, np.int64)
        self.tmp = np.zeros(n, np.int64)
        self.segbuf = np.zeros(OR_OPT_MAX_L + 1, np.int64)
        self.best = np.zeros(n, np.int64)
        self.saved = np.zeros(n, np.int64)
        self.ends = np.zeros(8, np.int64)
        self.todos = np.arange(n, dtype=np.int64)


def _pulir(tour, D, cand, ncand, ws):
    tour = tour.copy()
    pos = np.empty_like(tour)
    pos[tour] = np.arange(len(tour))
    _prefijos(tour, D, ws.F, ws.B)
    _busqueda_local(tour, pos, D, cand, ncand, ws.todos, len(tour), ws.qbuf, ws.inq,
                    ws.F, ws.B, ws.tmp, ws.segbuf, OR_OPT_MAX_L)
    return tour


def _ils(tour, D, cand, ncand, ws, limite_s, temp_rel):
    n = len(tour)
    tour = tour.copy()
    pos = np.empty_like(tour)
    pos[tour] = np.arange(n)
    cur = int(_largo(tour, D))
    if n < 8:
        return tour, cur, 0
    _prefijos(tour, D, ws.F, ws.B)
    window = max(1, min(max(DB_WINDOW_MIN, n // DB_WINDOW_DIV), (n - 2) // 3))
    limite_stall = max(STALL_MIN, STALL_FACTOR * n)
    ws.best[:] = tour
    best_len = cur
    temp = temp_rel * cur / n
    stall = 0
    kicks = 0
    fin = time.monotonic() + limite_s
    while stall < limite_stall and time.monotonic() < fin:
        bloque = min(ILS_BLOCK, limite_stall - stall)
        cur, best_len, stall = _ils_bloque(tour, pos, D, cand, ncand, cur, best_len, bloque, window,
                                           stall, temp, ws.qbuf, ws.inq, ws.F, ws.B, ws.tmp,
                                           ws.segbuf, OR_OPT_MAX_L, ws.best, ws.saved, ws.ends)
        cur, best_len, stall = int(cur), int(best_len), int(stall)
        kicks += bloque
    return ws.best.copy(), best_len, kicks


def resolver(D, xy=None, deposito=0, tiempo=10.0, semilla=None, retorno=True,
             temp=ACCEPT_TEMP, cand_m=CAND_M):
    """Resuelve la ruta que parte del depósito y visita todos los nodos.

    D: matriz n×n de costos enteros (p. ej. décimas de segundo), asimétrica, sin infinitos.
    xy: coordenadas proyectadas (opcional) para el arranque geométrico.
    retorno=False: ruta abierta (termina en la última parada, sin volver al depósito).
    Devuelve dict con 'orden' (empieza en el depósito), 'costo' y estadísticas.
    """
    D0 = np.ascontiguousarray(np.asarray(D, dtype=np.int64))
    n = D0.shape[0]
    if n < 2:
        return {'orden': list(range(n)), 'costo': 0, 'kicks': 0, 'arranques': {}}
    Dw = D0
    if not retorno:
        # Volver al depósito no cuesta: el ciclo óptimo es el camino abierto óptimo.
        Dw = D0.copy()
        Dw[:, deposito] = 0
    if semilla is not None:
        _semilla(int(semilla))
    rng = np.random.default_rng(semilla)
    t0 = time.monotonic()

    if n <= 3:
        orden = list(range(n))
    else:
        cand, ncand = candidatos(Dw, cand_m)
        ws = _Espacio(n)
        arranques = {'vecino': _vecino_mas_cercano(Dw, deposito)}
        if xy is not None:
            arranques['hilbert'] = _orden_hilbert(xy)
        for k in range(N_INSERCION):
            arranques[f'insercion{k}'] = _insercion(Dw, rng.permutation(n).astype(np.int64))
        # Embudo: búsqueda local completa a cada arranque; el mejor pasa al ILS.
        pulidos = {}
        for nombre, t in arranques.items():
            tp = _pulir(t, Dw, cand, ncand, ws)
            pulidos[nombre] = (int(_largo(tp, Dw)), tp)
        ganador = min(pulidos, key=lambda k: pulidos[k][0])
        restante = max(0.0, tiempo - (time.monotonic() - t0))
        tour, costo_w, kicks = _ils(pulidos[ganador][1], Dw, cand, ncand, ws, restante, temp)
        orden = list(map(int, tour))

    k = orden.index(deposito)
    orden = orden[k:] + orden[:k]
    if sorted(orden) != list(range(n)):
        raise RuntimeError('El tour no es una permutación válida')
    costo = costo_ruta(D0, orden, retorno)
    if n > 3:
        if costo_w != costo_ruta(Dw, orden, True):
            raise RuntimeError('El costo acumulado no coincide con el recalculado')
        info_arr = {k: v[0] for k, v in pulidos.items()}
    else:
        kicks, ganador, info_arr = 0, '-', {}
    return {'orden': orden, 'costo': int(costo), 'kicks': kicks, 'arranque': ganador,
            'arranques': info_arr, 'tiempo_s': time.monotonic() - t0}


def costo_ruta(D, orden, retorno=True):
    s = sum(int(D[orden[t], orden[t + 1]]) for t in range(len(orden) - 1))
    if retorno and len(orden) > 1:
        s += int(D[orden[-1], orden[0]])
    return s


def calentar():
    """Compila los kernels antes de medir tiempos."""
    rng = np.random.default_rng(0)
    D = rng.integers(1, 100, size=(30, 30)).astype(np.int64)
    resolver(D, xy=rng.random((30, 2)), tiempo=0.05, semilla=0)
