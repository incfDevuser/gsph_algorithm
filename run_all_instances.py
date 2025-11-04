import os
import time
import csv
from GSPH_BASE import gsph_fc, plot_gsph_fc, read_tsplib, RESULTS_DIR, BASE_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

instances_dir = os.path.abspath(BASE_DIR)
files = [f for f in os.listdir(instances_dir) if f.lower().endswith('.tsp')]
files.sort()

summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
with open(summary_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['instance', 'n_nodes', 'total_length', 'time_seconds', 'route_png', 'notes'])

for fname in files:
    inst_name = os.path.splitext(fname)[0]
    outdir = os.path.join(RESULTS_DIR, inst_name)
    os.makedirs(outdir, exist_ok=True)
    tsp_path = os.path.join(instances_dir, fname)
    try:
        nodes = read_tsplib(tsp_path)
    except Exception as e:
        with open(os.path.join(outdir, 'error.txt'), 'w', encoding='utf-8') as ef:
            ef.write(str(e))
        with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([inst_name, 0, '', '', os.path.join(outdir, 'error.txt'), 'read_error'])
        continue

    t0 = time.time()
    try:
        routes, conns, total_length, xm, ym = gsph_fc(nodes)
    except Exception as e:
        with open(os.path.join(outdir, 'error.txt'), 'w', encoding='utf-8') as ef:
            ef.write(str(e))
        with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([inst_name, len(nodes), '', '', os.path.join(outdir, 'error.txt'), 'run_error'])
        continue
    elapsed = time.time() - t0

    png_path = os.path.join(outdir, f"grafico_{inst_name}.png")
    plot_gsph_fc(routes, conns, xm, ym, save_path=png_path)

    res_txt = os.path.join(outdir, 'resultados.txt')
    with open(res_txt, 'w', encoding='utf-8') as f:
        f.write(f"Instancia: {fname}\n")
        f.write(f"N nodes: {len(nodes)}\n")
        f.write(f"Longitud total de la ruta: {total_length:.2f}\n")
        f.write(f"Tiempo de ejecución      : {elapsed:.3f} segundos\n")

    with open(summary_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([inst_name, len(nodes), f"{total_length:.2f}", f"{elapsed:.3f}", png_path, 'ok'])

    print(f"Processed {inst_name}: nodes={len(nodes)}, length={total_length:.2f}, time={elapsed:.2f}s")

print('\nAll done. Summary written to', summary_path)
