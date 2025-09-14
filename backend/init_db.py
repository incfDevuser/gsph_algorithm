from database import Base, engine
import models

print("Creando tablas")
Base.metadata.create_all(bind=engine)

