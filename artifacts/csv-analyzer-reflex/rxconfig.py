import os
import reflex as rx

PORT = int(os.environ.get("PORT", 3002))

config = rx.Config(
    app_name="csv_spike",
    port=PORT,
    backend_port=PORT + 1,
    # SPIKE FINDING: Reflex separa frontend (Next.js) e backend (FastAPI/WebSocket)
    # su due porte diverse. In Replit il proxy espone una porta per artifact.
    # Per il deploy su Replit servirà:
    #   - usare reflex run --env prod (compila frontend statico, backend lo serve)
    #   - oppure: nginx/reverse proxy interno per unificare le due porte
    # Questa configurazione funziona in sviluppo locale; per Replit prod vedere note spike.
)
