from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    character_id: str
    message: str


class EditRequest(BaseModel):
    panel_id: str
    mask_data: str  # base64 encoded PNG
    prompt: str


@router.get("/api/story")
async def get_story():
    """Get the current story state."""
    # TODO: return app.story_engine.story.to_dict()
    return {"status": "placeholder"}


@router.get("/api/pages")
async def get_pages():
    """Get all pages with panel layout data."""
    # TODO: return composed page layouts
    return {"pages": []}


@router.get("/api/pages/{page_number}")
async def get_page(page_number: int):
    """Get a specific page with full panel data."""
    # TODO: return comic_composer.compute_layout(page)
    return {"status": "placeholder", "page_number": page_number}


@router.post("/api/chat")
async def chat(message: ChatMessage):
    """Send a message to a character and get their response + new panels."""
    # TODO: story_engine.handle_player_input(message.character_id, message.message)
    return {"status": "placeholder"}


@router.post("/api/edit")
async def edit_panel(request: EditRequest):
    """Edit a panel region using AI inpainting."""
    # TODO: edit_engine.edit_panel_region(panel, mask, prompt)
    return {"status": "placeholder"}


@router.post("/api/animate/{panel_id}")
async def animate_panel(panel_id: str):
    """Generate an animated version of a panel."""
    # TODO: panel_generator.animate_panel(panel)
    return {"status": "placeholder"}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time generation progress updates."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # TODO: handle real-time communication
            await websocket.send_json({"status": "connected"})
    except WebSocketDisconnect:
        pass
