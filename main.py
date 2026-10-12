"""
GDARS v2 — Server Entry Point

Run locally:
    python main.py

Run with uvicorn directly:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

The reload flag is for development only — it restarts the server
on file changes. Never use --reload in production.
"""

import uvicorn
from config.settings import get_settings

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=(cfg.environment == "development"),
        workers=cfg.api.workers,
        log_level=cfg.api.log_level.lower(),
    )
