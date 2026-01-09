#!/usr/bin/env python3

import os
import sys
from PIL import Image
import argparse

def create_comparison_grid(instance_dir, output_name="comparacion_runs.png", max_cols=3):
    run_images = []
    for file in sorted(os.listdir(instance_dir)):
        if file.startswith('grafico_') and '_run_' in file and file.endswith('.png'):
            run_images.append(os.path.join(instance_dir, file))
    
    if not run_images:
        print(f"No se encontraron gráficos de runs en {instance_dir}")
        return None
    
    print(f"Encontrados {len(run_images)} gráficos de runs")
    images = [Image.open(img) for img in run_images]
    
    cell_width, cell_height = images[0].size
    
    num_images = len(images)
    cols = min(max_cols, num_images)
    rows = (num_images + cols - 1) // cols  # Redondear hacia arriba
    
    grid_width = cell_width * cols
    grid_height = cell_height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * cell_height
        grid_image.paste(img, (x, y))
    
    output_path = os.path.join(instance_dir, output_name)
    grid_image.save(output_path, quality=95)
    print(f"Comparación guardada en: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Genera una comparación visual de múltiples runs de una instancia TSP"
    )
    parser.add_argument(
        'instance_dir',
        help='Directorio con los resultados de la instancia (ej: ../results/berlin52)'
    )
    parser.add_argument(
        '-o', '--output',
        default='comparacion_runs.png',
        help='Nombre del archivo de salida (default: comparacion_runs.png)'
    )
    parser.add_argument(
        '-c', '--cols',
        type=int,
        default=3,
        help='Número de columnas en la cuadrícula (default: 3)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.instance_dir):
        print(f"Error: El directorio {args.instance_dir} no existe")
        return 1
    
    result = create_comparison_grid(args.instance_dir, args.output, args.cols)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
