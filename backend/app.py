import os
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.api import routes
from backend.api.routes import router
from backend.models.content_store import ContentStore
from backend.models.story import Story

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

app = FastAPI(title="Comiventure", version="0.1.0")
app.include_router(router)


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Disable caching for static files during development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Initialize content store and story
routes.content_store = ContentStore("data/content")
routes.story = Story("story-1", "Untitled Story")


@app.on_event("startup")
async def load_image_generator():
    """Load the image generator at startup if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            from backend.generator.image_generator import ImageGenerator
            routes.image_generator = ImageGenerator(routes.content_store)
            routes.image_generator.load_model()
            print(f"Image generator loaded on {torch.cuda.get_device_name(0)}")
        else:
            print("No CUDA GPU found — image generation disabled")
    except ImportError:
        print("PyTorch not installed — image generation disabled")


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


def _start_server():
    """Run the FastAPI server in a background thread."""
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


if __name__ == "__main__":
    if os.environ.get("COMIVENTURE_DESKTOP"):
        import webview

        server_thread = threading.Thread(target=_start_server, daemon=True)
        server_thread.start()

        window = webview.create_window(
            "Comiventure",
            f"http://127.0.0.1:{SERVER_PORT}",
            width=1280,
            height=800,
        )
        webview.start()
    else:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
