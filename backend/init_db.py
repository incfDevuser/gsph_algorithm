from database import Base, engine
import models

print("Creando tablas en PostgreSQL...")
Base.metadata.create_all(bind=engine)
print("¡Listo!")
