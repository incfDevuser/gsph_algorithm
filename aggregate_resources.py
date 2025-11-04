import os
import csv
import glob

ROOT = os.path.join(os.path.dirname(__file__), 'gsph_fc_results')
summary_in = os.path.join(ROOT, 'summary.csv')
summary_out = os.path.join(ROOT, 'resources_summary.csv')

instances = []
if os.path.isfile(summary_in):
    with open(summary_in, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append(row)
else:
    for p in os.listdir(ROOT):
        d = os.path.join(ROOT, p)
        if os.path.isdir(d):
            instances.append({'instance': p})

rows = []
for inst in instances:
    inst_name = inst.get('instance')
    folder = os.path.join(ROOT, inst_name)
    res_csv = os.path.join(folder, 'resources.csv')
    peak = ''
    avg_cpu = ''
    samples = ''
    if os.path.isfile(res_csv):
        with open(res_csv, 'r', encoding='utf-8') as rf:
            rreader = csv.DictReader(rf)
            rss_vals = []
            cpu_vals = []
            for r in rreader:
                try:
                    rss_vals.append(int(r['rss_bytes']))
                except Exception:
                    pass
                try:
                    cpu_vals.append(float(r['cpu_percent']))
                except Exception:
                    pass
            if rss_vals:
                peak = max(rss_vals)
            if cpu_vals:
                avg_cpu = sum(cpu_vals)/len(cpu_vals)
            samples = len(rss_vals)
    else:
        res_txt = os.path.join(folder, 'resultados.txt')
        if os.path.isfile(res_txt):
            with open(res_txt, 'r', encoding='utf-8') as rt:
                for line in rt:
                    if 'Peak RSS' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            try:
                                peak = int(parts[1].strip())
                            except Exception:
                                peak = parts[1].strip()
    row = {
        'instance': inst_name,
        'n_nodes': inst.get('n_nodes',''),
        'total_length': inst.get('total_length',''),
        'time_seconds': inst.get('time_seconds',''),
        'peak_rss_bytes': peak,
        'avg_cpu_percent': f"{avg_cpu:.2f}" if isinstance(avg_cpu, float) else avg_cpu,
        'samples': samples,
    }
    rows.append(row)

with open(summary_out, 'w', newline='', encoding='utf-8') as out:
    fieldnames = ['instance','n_nodes','total_length','time_seconds','peak_rss_bytes','avg_cpu_percent','samples']
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Wrote', summary_out)
