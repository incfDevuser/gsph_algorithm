from gsph_genetic import optimize_route_genetic

depot = {"id": "D1", "name": "Depot Central", "lat": -33.447487, "lng": -70.673676}

orders = [
    {"id": "O1", "lat": -33.441000, "lng": -70.680000},
    {"id": "O2", "lat": -33.450000, "lng": -70.670000},
    {"id": "O3", "lat": -33.455000, "lng": -70.665000},
]

print("Ejecutando optimización genética GSPH (test simple)...")
try:
    coords, length, sequence = optimize_route_genetic(
        depot, orders, 
        population_size=20, 
        generations=10
    )
    
    print(f"\nLongitud final del tour: {length:.2f}")
    print("Coordenadas optimizadas:")
    for i, (lat, lng) in enumerate(coords):
        print(f"  {i+1}: ({lat:.6f}, {lng:.6f})")
    
    print("\nSecuencia de órdenes optimizada:")
    print(f"  {sequence}")
    
    print("\nTour lógico:")
    print("  Depot →", " → ".join(sequence), "→ Depot")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()