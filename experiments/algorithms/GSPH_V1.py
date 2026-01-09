import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import time
import os

#EPS_FRONTIER   = 5
#MAX_ITER_LOCAL = 800
#RESULTS_DIR = "gsph_fc_results"

def read_tsplib(filename):
    nodes = []
    with open(filename, 'r') as f:
        reading_nodes = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if line == "EOF":
                break
            if reading_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    nodes.append((x, y))
    return nodes

def euclidean_distance(p1, p2):
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return int(dist + 0.5)

def total_path_length(path):
    return sum(euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1))

def tsp_2opt(points, max_iter=800):
    def two_opt_swap(r, i, k):
        return r[:i] + r[i:k+1][::-1] + r[k+1:]
    best = points[:]
    best_dist = total_path_length(best)
    changed, it = True, 0
    while changed and it < max_iter:
        changed = False
        for i in range(1, len(points)-2):
            for k in range(i+1, len(points)):
                new = two_opt_swap(best, i, k)
                new_dist = total_path_length(new)
                if new_dist < best_dist:
                    best, best_dist, changed = new, new_dist, True
        it += 1
    return best

def subdivide_quadrants(pts):
    xs, ys = zip(*pts)
    xmid, ymid = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    quads = {'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]}
    for p in pts:
        x, y = p
        if x <= xmid and y >  ymid: quads['Q1'].append(p)
        elif x >  xmid and y >  ymid: quads['Q2'].append(p)
        elif x <= xmid and y <= ymid: quads['Q3'].append(p)
        else: quads['Q4'].append(p)
    return quads, xmid, ymid

def best_frontier_pair(A, B, direction, mid, eps):
    filt = (lambda p: abs(p[0]-mid)<eps) if direction=='vertical' else (lambda p: abs(p[1]-mid)<eps)
    candA = [p for p in A if filt(p)]
    candB = [p for p in B if filt(p)]
    best, best_d = (None,None), float('inf')
    for a in candA:
        for b in candB:
            d = euclidean_distance(a,b)
            if d < best_d:
                best, best_d = (a,b), d
    return best, best_d

def gsph_fc(nodes,MAX_ITER_LOCAL=800 ,EPS_FRONTIER = 5):
    quads, xmid, ymid = subdivide_quadrants(nodes)
    routes = {}
    for q, pts in quads.items():
        #print(q, pts)
        if len(pts) > 1:
            rt = tsp_2opt(pts, MAX_ITER_LOCAL)
            routes[q] = rt
        else:
            routes[q] = pts
    neighbor_pairs = [
        ('Q1','Q2','vertical',  xmid),
        ('Q1','Q3','horizontal',ymid),
        ('Q2','Q4','horizontal',ymid),
        ('Q3','Q4','vertical',  xmid)
    ]
    connections = []
    inter_len   = 0
    for q1,q2,dirc,mid in neighbor_pairs:
        (a,b), d = best_frontier_pair(quads[q1], quads[q2], dirc, mid, EPS_FRONTIER)
        if a and b:
            connections.append((a,b))
            inter_len += d
    sub_len = sum(total_path_length(r) for r in routes.values())
    print(sub_len)
    #print(total_length)
    return routes, connections, sub_len+inter_len, xmid, ymid

def recoverTour(routes,problem):
    tour = []
    for j in routes:
        for k in routes[j]:
            for i in range(problem.dimension):
                tmpNode = problem.node_coords[i+1]
                compareNode = (tmpNode[0], tmpNode[1])
                if(k == compareNode):
                    tour.append(i+1)
    return tour
    


def plot_gsph_fc(routes, conns, xmid, ymid, save_path=None):
    plt.figure(figsize=(8,8))
    colors = ['blue', 'green', 'red', 'orange']
    for idx, (q, r) in enumerate(routes.items()):
        if len(r) > 1:
            x, y = zip(*r)
            plt.plot(x, y, marker='o', color=colors[idx])
        else:
            plt.scatter(*r[0], color='black', marker='x')
    subroutes = list(routes.values())
    for i in range(len(subroutes)-1):
        last_point = subroutes[i][-1]
        next_point = subroutes[i+1][0]
        plt.plot([last_point[0], next_point[0]], [last_point[1], next_point[1]],
                 linestyle='--', color='black', linewidth=2)
    last_to_first = (subroutes[-1][-1], subroutes[0][0])
    plt.plot([last_to_first[0][0], last_to_first[1][0]],
             [last_to_first[0][1], last_to_first[1][1]],
             linestyle='--', color='black', linewidth=2)
    plt.axvline(xmid, linestyle='--', color='blue')
    plt.axhline(ymid, linestyle='--', color='blue')
    plt.title("GSPH–FC aplicado a instancia TSPLIB")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    nodes = read_tsplib('a280.tsp')
    t0 = time.time()
    routes, conns, total_length, xm, ym = gsph_fc(nodes)
    t_heur = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "resultados.txt"), "w") as f:
        f.write("── RESULTADOS GSPH–FC ──\n")
        f.write(f"Longitud total de la ruta: {total_length:.2f}\n")
        f.write(f"Tiempo de ejecución      : {t_heur:.3f} segundos\n")
    print("\n── RESULTADOS GSPH–FC ──")
    print(f"Longitud total de la ruta: {total_length:.2f}")
    print(f"Tiempo de ejecución      : {t_heur:.3f} segundos")
    plot_gsph_fc(routes, conns, xm, ym, save_path=os.path.join(RESULTS_DIR, "grafico_gsph_fc.png"))