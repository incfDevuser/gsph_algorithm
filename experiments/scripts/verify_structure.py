#!/usr/bin/env python3
"""
Script de verificación de la estructura del proyecto GSPH.
Verifica que todos los archivos y carpetas necesarios estén en su lugar.
"""
import os
import sys

def check_structure():
    """Verifica la estructura del proyecto."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    required_structure = {
        'experiments': {
            'algorithms': ['GSPH_BASE.py', 'GSPH_V1.py'],
            'instances': {
                'standard': [],
                'large': []
            },
            'benchmarks': ['BKS'],
            'results': [],
            'scripts': ['run_all_instances.py', 'aggregate_resources.py'],
            'docs': ['apuntes.txt', 'TEXT.txt']
        },
        'backend': [],
        'frontend': []
    }
    
    print("🔍 Verificando estructura del proyecto GSPH...\n")
    
    all_ok = True
    
    def check_path(base, structure, indent=0):
        nonlocal all_ok
        for name, content in structure.items():
            path = os.path.join(base, name)
            prefix = "  " * indent
            
            if isinstance(content, dict):
                if os.path.isdir(path):
                    print(f"{prefix}✅ {name}/")
                    check_path(path, content, indent + 1)
                else:
                    print(f"{prefix}❌ {name}/ (falta)")
                    all_ok = False
            elif isinstance(content, list):
                if os.path.isdir(path):
                    print(f"{prefix}✅ {name}/")
                    for file in content:
                        file_path = os.path.join(path, file)
                        if os.path.exists(file_path):
                            print(f"{prefix}  ✅ {file}")
                        else:
                            print(f"{prefix}  ⚠️  {file} (no encontrado, puede ser opcional)")
                else:
                    print(f"{prefix}❌ {name}/ (falta)")
                    all_ok = False
    
    check_path(project_root, required_structure)
    
    # Verificar archivos importantes en la raíz
    print("\n📄 Archivos en raíz:")
    root_files = ['README.md', 'requirements.txt', '.gitignore']
    for file in root_files:
        path = os.path.join(project_root, file)
        if os.path.exists(path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (falta)")
            all_ok = False
    
    print("\n" + "="*50)
    if all_ok:
        print("✅ Estructura del proyecto correcta!")
        print("\n🚀 Puedes ejecutar el algoritmo con:")
        print("   cd experiments/algorithms")
        print("   python3 GSPH_BASE.py berlin52")
    else:
        print("⚠️  Algunos elementos están faltantes")
        print("   Verifica la estructura del proyecto")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(check_structure())
