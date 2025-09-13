"""
Script de prueba para comparar GSPH original vs GSPH Genético
"""

import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gsph_algorithm import optimize_route
from gsph_genetic import optimize_route_genetic

def load_test_data():
    """Cargar datos de prueba"""
    json_path = "../frontend/src/data/test_orders_genetic.json"
    if not os.path.exists(json_path):
        depot = {"id": "D1", "name": "Depot Central", "lat": -33.447487, "lng": -70.673676}
        orders = [
            {"id": "O1", "lat": -33.441000, "lng": -70.680000},
            {"id": "O2", "lat": -33.450000, "lng": -70.670000},
            {"id": "O3", "lat": -33.455000, "lng": -70.665000},
            {"id": "O4", "lat": -33.445000, "lng": -70.685000},
            {"id": "O5", "lat": -33.440000, "lng": -70.675000},
            {"id": "O6", "lat": -33.452000, "lng": -70.668000},
            {"id": "O7", "lat": -33.448000, "lng": -70.672000},
            {"id": "O8", "lat": -33.443000, "lng": -70.678000},
        ]
        return depot, orders
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['depot'], data['orders']

def run_comparison():
    """Ejecutar comparación entre métodos"""
    print("=" * 80)
    print("COMPARACIÓN: GSPH Original vs GSPH Genético")
    print("=" * 80)
    
    depot, orders = load_test_data()
    print(f"Depot: {depot['name']}")
    print(f"Número de órdenes: {len(orders)}")
    
    print("\n" + "-" * 40)
    print("EJECUTANDO GSPH ORIGINAL")
    print("-" * 40)
    
    start_time = time.time()
    coords_original, length_original = optimize_route(depot, orders, method="gsph")
    time_original = time.time() - start_time
    
    print(f"Tiempo de ejecución: {time_original:.2f} segundos")
    print(f"Longitud del tour: {length_original:.2f}")
    print(f"Número de puntos: {len(coords_original)}")
    
    print("\n" + "-" * 40)
    print("EJECUTANDO GSPH GENÉTICO")
    print("-" * 40)
    
    start_time = time.time()
    coords_genetic, length_genetic = optimize_route_genetic(
        depot, orders, 
        population_size=50, 
        generations=100
    )
    time_genetic = time.time() - start_time
    
    print(f"Tiempo de ejecución: {time_genetic:.2f} segundos")
    print(f"Longitud del tour: {length_genetic:.2f}")
    print(f"Número de puntos: {len(coords_genetic)}")
    
    print("\n" + "=" * 40)
    print("RESUMEN COMPARATIVO")
    print("=" * 40)
    
    improvement = ((length_original - length_genetic) / length_original) * 100
    time_ratio = time_genetic / time_original
    
    print(f"GSPH Original:")
    print(f"  - Tiempo: {time_original:.2f}s")
    print(f"  - Longitud: {length_original:.2f}")
    
    print(f"\nGSPH Genético:")
    print(f"  - Tiempo: {time_genetic:.2f}s") 
    print(f"  - Longitud: {length_genetic:.2f}")
    
    print(f"\nComparación:")
    if improvement > 0:
        print(f"  - Mejora: {improvement:.2f}% (Genético es mejor)")
    else:
        print(f"  - Diferencia: {abs(improvement):.2f}% (Original es mejor)")
    
    print(f"  - Ratio de tiempo: {time_ratio:.2f}x")
    results = {
        "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "depot": depot,
        "orders_count": len(orders),
        "gsph_original": {
            "execution_time": time_original,
            "tour_length": length_original,
            "coordinates": coords_original
        },
        "gsph_genetic": {
            "execution_time": time_genetic,
            "tour_length": length_genetic,
            "coordinates": coords_genetic
        },
        "improvement_percentage": improvement,
        "time_ratio": time_ratio
    }
    
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Resultados guardados en 'comparison_results.json'")
    
    return results

def run_genetic_parameter_test():
    """Probar diferentes configuraciones del algoritmo genético"""
    print("\n" + "=" * 80)
    print("PRUEBA DE PARÁMETROS GENÉTICOS")
    print("=" * 80)
    
    depot, orders = load_test_data()
    configs = [
        {"pop_size": 30, "generations": 50, "name": "Rápido"},
        {"pop_size": 50, "generations": 100, "name": "Balanceado"},
        {"pop_size": 100, "generations": 200, "name": "Intensivo"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Configuración {config['name']} ---")
        print(f"Población: {config['pop_size']}, Generaciones: {config['generations']}")
        
        start_time = time.time()
        coords, length = optimize_route_genetic(
            depot, orders,
            population_size=config['pop_size'],
            generations=config['generations']
        )
        execution_time = time.time() - start_time
        
        result = {
            "config": config['name'],
            "population_size": config['pop_size'],
            "generations": config['generations'],
            "execution_time": execution_time,
            "tour_length": length
        }
        results.append(result)
        
        print(f"Tiempo: {execution_time:.2f}s, Longitud: {length:.2f}")
    
    best_result = min(results, key=lambda x: x['tour_length'])
    
    print(f"\n Mejor configuración: {best_result['config']}")
    print(f"   Longitud: {best_result['tour_length']:.2f}")
    print(f"   Tiempo: {best_result['execution_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    try:
        main_results = run_comparison()
        
        param_results = run_genetic_parameter_test()
        
        print("Listo")
        
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {str(e)}")
        import traceback
        traceback.print_exc()