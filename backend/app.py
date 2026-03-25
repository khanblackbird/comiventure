import threading

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.api.routes import router

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

app = FastAPI(title="Comiventure", version="0.1.0")
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


def _start_server():
    """Run the FastAPI server in a background thread."""
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    window = webview.create_window(
        "Comiventure",
        f"http://{SERVER_HOST}:{SERVER_PORT}",
        width=1280,
        height=800,
    )
    webview.start()
