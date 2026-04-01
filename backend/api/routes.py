"""API routes — enforce the hierarchy.

Every operation goes through the Story object. You can't generate
an image without a complete chain:
  Character -> Chapter -> Page -> Panel -> Script

The API refuses to operate on incomplete data.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from backend.models import Story, Character, Chapter, Page, Panel, Script, ContentStore

router = APIRouter()

# Set by app.py at startup
image_generator = None
content_store: ContentStore | None = None
story: Story | None = None

# One generation at a time — 8GB VRAM
_generation_lock = asyncio.Lock()


# --- Request models ---

class CreateCharacterRequest(BaseModel):
    name: str
    description: str = ""
    personality_prompt: str = ""
    appearance_prompt: str = ""
    is_temporary: bool = False


class UpdateCharacterRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    personality_prompt: Optional[str] = None
    appearance_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None


class CreateChapterRequest(BaseModel):
    title: str
    synopsis: str = ""
    character_ids: list[str]
    default_location: str = ""
    default_time_of_day: str = ""


class UpdateChapterRequest(BaseModel):
    title: Optional[str] = None
    synopsis: Optional[str] = None
    default_location: Optional[str] = None
    default_time_of_day: Optional[str] = None
    negative_prompt: Optional[str] = None


class CreatePageRequest(BaseModel):
    chapter_id: str
    layout_template: str = "auto"


class UpdatePageRequest(BaseModel):
    setting: Optional[str] = None
    mood: Optional[str] = None
    action_context: Optional[str] = None
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    lighting: Optional[str] = None
    negative_prompt: Optional[str] = None


class CreatePanelRequest(BaseModel):
    page_id: str


class UpdatePanelRequest(BaseModel):
    shot_type: Optional[str] = None
    narration: Optional[str] = None
    negative_prompt: Optional[str] = None


class CreateScriptRequest(BaseModel):
    panel_id: str
    character_id: str
    dialogue: str = ""
    action: str = ""
    direction: str = ""
    emotion: str = ""
    pose: str = ""
    outfit: str = ""


class UpdateScriptRequest(BaseModel):
    dialogue: Optional[str] = None
    action: Optional[str] = None
    direction: Optional[str] = None
    emotion: Optional[str] = None
    pose: Optional[str] = None
    outfit: Optional[str] = None
    negative_prompt: Optional[str] = None
    source: str = "manual"


class UpdateAppearanceRequest(BaseModel):
    species: Optional[str] = None
    body_type: Optional[str] = None
    height: Optional[str] = None
    skin_tone: Optional[str] = None
    hair_style: Optional[str] = None
    hair_colour: Optional[str] = None
    eye_colour: Optional[str] = None
    facial_features: Optional[str] = None
    outfit: Optional[str] = None
    accessories: Optional[str] = None
    art_style_notes: Optional[str] = None


class AddReferenceRequest(BaseModel):
    content_hash: str
    source: str = "upload"
    caption: str = ""
    pose: str = ""
    expression: str = ""
    angle: str = ""
    scene: str = ""
    outfit_variant: str = ""
    tags: list[str] = []


class UpdateReferenceRequest(BaseModel):
    caption: Optional[str] = None
    pose: Optional[str] = None
    expression: Optional[str] = None
    angle: Optional[str] = None
    scene: Optional[str] = None
    outfit_variant: Optional[str] = None
    tags: Optional[list[str]] = None
    accepted: Optional[bool] = None


class InpaintRequest(BaseModel):
    panel_id: str
    mask_data: str  # base64 encoded PNG mask
    prompt: str
    negative_prompt: Optional[str] = None
    strength: float = 0.75
    steps: int = 25
    seed: Optional[int] = None


class GeneratePanelRequest(BaseModel):
    panel_id: str
    negative_prompt: Optional[str] = None
    width: int = 768
    height: int = 512
    steps: int = 25
    guidance_scale: float = 7.0
    seed: Optional[int] = None


# --- Story ---

STORIES_DIR = Path("data/stories")
STORIES_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/api/story")
async def get_story():
    _require_story()
    return story.to_dict()


@router.post("/api/story/new")
async def new_story(title: str = "Untitled Story"):
    """Create a fresh empty story. Frontend bootstraps characters/chapters."""
    global story
    from backend.models.ids import make_id
    story = Story(make_id("story"), title)
    return story.to_dict()


class UpdateStoryRequest(BaseModel):
    title: Optional[str] = None
    synopsis: Optional[str] = None
    art_style: Optional[str] = None
    genre: Optional[str] = None
    negative_prompt: Optional[str] = None


@router.put("/api/story")
async def update_story(request: UpdateStoryRequest):
    """Update story properties — title, synopsis, art style, genre."""
    _require_story()
    if request.title is not None:
        story.title = request.title
    if request.synopsis is not None:
        story.synopsis = request.synopsis
    if request.art_style is not None:
        story.art_style = request.art_style
    if request.genre is not None:
        story.genre = request.genre
    if request.negative_prompt is not None:
        story.negative_prompt = request.negative_prompt
    story.emit("story_updated", story)
    return story.to_dict()


@router.post("/api/story/save")
async def save_story_endpoint():
    """Save the current story to a .cvn file. Returns the filename."""
    _require_story()
    if not content_store:
        raise HTTPException(503, "Content store not available")

    from backend.models.storage import save_story as do_save
    filename = f"{story.title.replace(' ', '_').lower()}_{story.story_id}.cvn"
    filepath = str(STORIES_DIR / filename)
    do_save(story, content_store, filepath)
    return {"filename": filename, "path": filepath}


@router.post("/api/story/load")
async def load_story_endpoint(file: UploadFile = File(...)):
    """Load a story from an uploaded .cvn file."""
    global story
    if not content_store:
        raise HTTPException(503, "Content store not available")

    import tempfile
    from backend.models.storage import load_story as do_load

    with tempfile.NamedTemporaryFile(suffix=".cvn", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        story = do_load(tmp_path, content_store)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return story.to_dict()


@router.get("/api/stories")
async def list_saved_stories():
    """List all saved .cvn files."""
    stories = []
    for cvn_file in STORIES_DIR.glob("*.cvn"):
        stories.append({
            "filename": cvn_file.name,
            "size_bytes": cvn_file.stat().st_size,
            "modified": cvn_file.stat().st_mtime,
        })
    stories.sort(key=lambda s: s["modified"], reverse=True)
    return stories


@router.post("/api/story/load/{filename}")
async def load_saved_story(filename: str):
    """Load a previously saved story by filename."""
    global story
    if not content_store:
        raise HTTPException(503, "Content store not available")

    filepath = STORIES_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"Story file '{filename}' not found")

    from backend.models.storage import load_story as do_load
    story = do_load(str(filepath), content_store)
    return story.to_dict()


@router.get("/api/story/download")
async def download_story():
    """Download the current story as a .cvn file."""
    _require_story()
    if not content_store:
        raise HTTPException(503, "Content store not available")

    from backend.models.storage import save_story as do_save
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cvn", delete=False) as tmp:
        tmp_path = tmp.name
    do_save(story, content_store, tmp_path)
    return FileResponse(tmp_path, filename=f"{story.title}.cvn", media_type="application/zip")


@router.post("/api/story/import-character")
async def import_character(file: UploadFile = File(...)):
    """Import a character from another story's .cvn file.
    Loads the file, extracts all characters, adds them to the current story.
    """
    _require_story()
    if not content_store:
        raise HTTPException(503, "Content store not available")

    import tempfile
    from backend.models.storage import load_story as do_load

    with tempfile.NamedTemporaryFile(suffix=".cvn", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        source_story = do_load(tmp_path, content_store)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    imported = []
    for char_id, character in source_story.characters.items():
        if char_id not in story.characters:
            from backend.models import Character
            new_char = Character(
                character_id=character.character_id,
                name=character.name,
                description=character.description,
                personality_prompt=character.personality_prompt,
                appearance_prompt=character.appearance_prompt,
                is_temporary=character.is_temporary,
            )
            new_char.appearance = character.appearance
            story.add_character(new_char)
            imported.append(new_char.to_dict())

    return {"imported": imported, "count": len(imported)}


# --- LoRA Library ---

LORA_DIR = Path("data/loras")
LORA_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/api/loras")
async def list_loras():
    """List all LoRA files in the local library."""
    loras = []
    for path in LORA_DIR.glob("*.safetensors"):
        loras.append({
            "filename": path.name,
            "name": path.stem,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        })
    # Also include any LoRAs stored in content store (from .cvn loads)
    if story:
        for lora in story.style_loras:
            if not any(l["name"] == lora.get("name") for l in loras):
                loras.append(lora)
    return {"loras": loras}


@router.post("/api/loras/upload")
async def upload_lora(file: UploadFile = File(...)):
    """Upload a .safetensors LoRA file to the local library."""
    if not file.filename or not file.filename.endswith(".safetensors"):
        raise HTTPException(400, "File must be a .safetensors file")

    lora_bytes = await file.read()
    lora_path = LORA_DIR / file.filename
    lora_path.write_bytes(lora_bytes)

    # Also store in content store so it saves with .cvn
    content_hash = None
    if content_store:
        content_hash = content_store.store(
            lora_bytes, "application/octet-stream",
            metadata={"type": "lora", "filename": file.filename},
        )

    return {
        "filename": file.filename,
        "name": lora_path.stem,
        "size_mb": round(len(lora_bytes) / (1024 * 1024), 1),
        "content_hash": content_hash,
    }


@router.post("/api/story/loras")
async def set_story_loras(request: dict):
    """Set the active LoRAs for the current story.
    Body: {"loras": [{"name": "...", "filename": "...", "strength": 0.7, "content_hash": "..."}]}
    """
    _require_story()
    story.style_loras = request.get("loras", [])

    # Load into pipeline if available
    if image_generator and image_generator.pipeline:
        try:
            # Unload existing LoRAs first
            image_generator.pipeline.unload_lora_weights()
        except Exception:
            pass

        for lora in story.style_loras:
            lora_path = LORA_DIR / lora.get("filename", "")
            if lora_path.exists():
                try:
                    adapter_name = lora.get("name", lora_path.stem)
                    image_generator.pipeline.load_lora_weights(
                        str(LORA_DIR),
                        weight_name=lora_path.name,
                        adapter_name=adapter_name,
                        local_files_only=True,
                    )
                    scale = lora.get("strength", 0.7)
                    image_generator.pipeline.set_adapters(
                        [adapter_name], adapter_weights=[scale],
                    )
                    print(f"Loaded LoRA: {adapter_name} at {scale}")
                except Exception as e:
                    print(f"Failed to load LoRA {lora_path.name}: {e}")

    return {"active_loras": story.style_loras}


# --- Style References (IP-Adapter) ---

@router.get("/api/story/style-references")
async def get_style_references():
    """Get the story's style reference images for IP-Adapter conditioning."""
    _require_story()
    return {
        "references": story.style_references,
        "urls": [f"/api/content/{h}" for h in story.style_references],
    }


@router.post("/api/story/style-references/upload")
async def upload_style_reference(file: UploadFile = File(...)):
    """Upload a style reference image for story-level IP-Adapter conditioning."""
    _require_story()
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = await file.read()
    content_hash = content_store.store(
        image_bytes, file.content_type or "image/png",
        metadata={"type": "style_reference"},
    )
    if content_hash not in story.style_references:
        story.style_references.append(content_hash)

    return {
        "content_hash": content_hash,
        "image_url": f"/api/content/{content_hash}",
        "total_references": len(story.style_references),
    }


@router.delete("/api/story/style-references/{content_hash}")
async def remove_style_reference(content_hash: str):
    """Remove a style reference image."""
    _require_story()
    story.style_references = [
        h for h in story.style_references if h != content_hash
    ]
    return {"remaining": len(story.style_references)}


# --- Civitai LoRA Browser ---

@router.get("/api/civitai/search")
async def civitai_search(
    query: str = "",
    tag: str = "",
    page: int = 1,
    limit: int = 20,
):
    """Search civitai for SDXL LoRA models.
    Returns previews, metadata, and download URLs.
    """
    import httpx

    params = {
        "types": "LORA",
        "sort": "Highest Rated",
        "period": "AllTime",
        "limit": limit,
        "page": page,
        "baseModels": "SDXL 1.0",
    }
    if query:
        params["query"] = query
    if tag:
        params["tag"] = tag

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://civitai.com/api/v1/models",
                params=params,
                timeout=15.0,
            )
            if response.status_code != 200:
                raise HTTPException(502, f"Civitai returned {response.status_code}")

            data = response.json()
            results = []
            for model in data.get("items", []):
                # Get the latest SDXL LoRA version
                version = None
                for v in model.get("modelVersions", []):
                    if "SDXL 1.0" in (v.get("baseModel", "") or ""):
                        version = v
                        break
                if not version:
                    version = model.get("modelVersions", [{}])[0]

                # Get preview image
                preview_url = ""
                for img in version.get("images", []):
                    preview_url = img.get("url", "")
                    break

                # Get download URL and file info
                download_url = ""
                filename = ""
                size_mb = 0
                for f in version.get("files", []):
                    if f.get("name", "").endswith(".safetensors"):
                        download_url = f.get("downloadUrl", "")
                        filename = f.get("name", "")
                        size_mb = round(
                            f.get("sizeKB", 0) / 1024, 1
                        )
                        break

                results.append({
                    "id": model.get("id"),
                    "name": model.get("name", ""),
                    "description": (
                        model.get("description", "")[:200]
                    ),
                    "tags": model.get("tags", []),
                    "rating": model.get("stats", {}).get("rating", 0),
                    "downloads": model.get("stats", {}).get(
                        "downloadCount", 0
                    ),
                    "preview_url": preview_url,
                    "download_url": download_url,
                    "filename": filename,
                    "size_mb": size_mb,
                    "version_name": version.get("name", ""),
                })

            return {
                "results": results,
                "total": data.get("metadata", {}).get(
                    "totalItems", 0
                ),
                "page": page,
            }

    except httpx.HTTPError as e:
        raise HTTPException(502, f"Civitai request failed: {e}")


@router.post("/api/civitai/download")
async def civitai_download(request: dict):
    """Download a LoRA from civitai by URL and add to library."""
    download_url = request.get("download_url")
    filename = request.get("filename", "model.safetensors")

    if not download_url:
        raise HTTPException(400, "download_url required")
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"

    import httpx

    lora_path = LORA_DIR / filename
    if lora_path.exists():
        return {
            "filename": filename,
            "name": lora_path.stem,
            "size_mb": round(lora_path.stat().st_size / (1024 * 1024), 1),
            "status": "already_exists",
        }

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(download_url, timeout=120.0)
            if response.status_code != 200:
                raise HTTPException(
                    502, f"Download failed: {response.status_code}"
                )

            lora_bytes = response.content
            lora_path.write_bytes(lora_bytes)

            # Store in content store for .cvn save
            content_hash = None
            if content_store:
                content_hash = content_store.store(
                    lora_bytes, "application/octet-stream",
                    metadata={
                        "type": "lora",
                        "filename": filename,
                    },
                )

            return {
                "filename": filename,
                "name": lora_path.stem,
                "size_mb": round(
                    len(lora_bytes) / (1024 * 1024), 1
                ),
                "content_hash": content_hash,
                "status": "downloaded",
            }

    except httpx.HTTPError as e:
        raise HTTPException(502, f"Download failed: {e}")


@router.get("/api/story/validate")
async def validate_story():
    """Check graph integrity. Returns list of violations (empty = valid)."""
    _require_story()
    errors = story.validate()
    return {"valid": len(errors) == 0, "errors": errors}


# --- Characters ---

@router.get("/api/characters")
async def list_characters():
    _require_story()
    return {
        character_id: character.to_dict()
        for character_id, character in story.characters.items()
    }


@router.post("/api/characters")
async def create_character(request: CreateCharacterRequest):
    _require_story()
    import uuid
    character_id = f"char-{uuid.uuid4().hex[:8]}"
    character = Character(
        character_id=character_id,
        name=request.name,
        description=request.description,
        personality_prompt=request.personality_prompt,
        appearance_prompt=request.appearance_prompt,
        is_temporary=request.is_temporary,
    )
    story.add_character(character)
    return character.to_dict()


@router.put("/api/characters/{character_id}")
async def update_character(character_id: str, request: UpdateCharacterRequest):
    character = _require_character(character_id)
    character.update(
        name=request.name,
        description=request.description,
        personality_prompt=request.personality_prompt,
        appearance_prompt=request.appearance_prompt,
    )
    if request.negative_prompt is not None:
        character.negative_prompt = request.negative_prompt
    return character.to_dict()


@router.delete("/api/characters/{character_id}")
async def delete_character(character_id: str):
    _require_character(character_id)
    story.remove_character(character_id)
    return {"deleted": character_id}


# --- Appearance ---

@router.get("/api/characters/{character_id}/appearance")
async def get_appearance(character_id: str):
    character = _require_character(character_id)
    return character.appearance.to_dict()


@router.put("/api/characters/{character_id}/appearance")
async def update_appearance(character_id: str, request: UpdateAppearanceRequest):
    character = _require_character(character_id)
    properties = {key: value for key, value in request.model_dump().items() if value is not None}
    character.update_appearance(properties=properties)
    return character.appearance.to_dict()


@router.post("/api/characters/{character_id}/references/upload")
async def upload_reference(
    character_id: str,
    file: UploadFile = File(...),
    caption: str = Form(""),
    pose: str = Form(""),
    expression: str = Form(""),
    angle: str = Form(""),
    scene: str = Form(""),
    outfit_variant: str = Form(""),
):
    """Upload an image file directly into the character's reference bank."""
    character = _require_character(character_id)
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = await file.read()
    content_type = file.content_type or "image/png"
    content_hash = content_store.store(image_bytes, content_type)

    reference = character.appearance.add_reference(
        content_hash=content_hash,
        source="upload",
        caption=caption,
        pose=pose,
        expression=expression,
        angle=angle,
        scene=scene,
        outfit_variant=outfit_variant,
    )
    character.emit("character_updated", character)
    return reference.to_dict()


@router.post("/api/characters/{character_id}/references")
async def add_reference(character_id: str, request: AddReferenceRequest):
    """Add an existing content hash as a reference (e.g. from a generated/edited panel)."""
    character = _require_character(character_id)
    if not content_store or not content_store.exists(request.content_hash):
        raise HTTPException(404, "Content hash not found in store")

    reference = character.appearance.add_reference(
        content_hash=request.content_hash,
        source=request.source,
        caption=request.caption,
        pose=request.pose,
        expression=request.expression,
        angle=request.angle,
        scene=request.scene,
        outfit_variant=request.outfit_variant,
        tags=request.tags,
    )
    character.emit("character_updated", character)
    return reference.to_dict()


@router.get("/api/characters/{character_id}/references")
async def list_references(character_id: str):
    character = _require_character(character_id)
    return [ref.to_dict() for ref in character.appearance.references]


@router.put("/api/characters/{character_id}/references/{content_hash}")
async def update_reference(character_id: str, content_hash: str, request: UpdateReferenceRequest):
    """Update labels/rating on a reference image."""
    character = _require_character(character_id)
    ref = character.appearance.get_reference(content_hash)
    if not ref:
        raise HTTPException(404, f"Reference {content_hash} not found")

    if request.caption is not None:
        ref.caption = request.caption
    if request.pose is not None:
        ref.pose = request.pose
    if request.expression is not None:
        ref.expression = request.expression
    if request.angle is not None:
        ref.angle = request.angle
    if request.scene is not None:
        ref.scene = request.scene
    if request.outfit_variant is not None:
        ref.outfit_variant = request.outfit_variant
    if request.tags is not None:
        ref.tags = request.tags
    if request.accepted is not None:
        ref.accepted = request.accepted

    character.emit("character_updated", character)
    return ref.to_dict()


@router.delete("/api/characters/{character_id}/references/{content_hash}")
async def delete_reference(character_id: str, content_hash: str):
    character = _require_character(character_id)
    character.appearance.remove_reference(content_hash)
    character.emit("character_updated", character)
    return {"deleted": content_hash}


@router.get("/api/characters/{character_id}/training-pairs")
async def get_training_pairs(character_id: str):
    """Get accepted reference images as training pairs (hash, caption)."""
    character = _require_character(character_id)
    pairs = character.appearance.training_pairs()
    return {
        "character_id": character_id,
        "count": len(pairs),
        "pairs": [{"content_hash": h, "caption": c} for h, c in pairs],
    }


# --- Character export/import ---

@router.get("/api/characters/{character_id}/export")
async def export_character(character_id: str):
    """Export a character as a standalone package (JSON + reference image hashes).
    Use with content endpoint to download the actual images.
    """
    character = _require_character(character_id)
    return {
        "character": character.to_dict(),
        "appearance": character.appearance.to_dict(),
        "reference_urls": {
            ref.content_hash: f"/api/content/{ref.content_hash}"
            for ref in character.appearance.references
        },
    }


# --- Profile ---

class UpdateProfileRequest(BaseModel):
    biography: Optional[str] = None
    personality: Optional[str] = None
    physical: Optional[dict] = None
    tendencies: Optional[list[str]] = None
    expressions: Optional[dict[str, str]] = None
    notes: Optional[str] = None


class AddOutfitRequest(BaseModel):
    name: str
    description: str
    is_default: bool = False


class SetRelationshipRequest(BaseModel):
    target_character_id: str
    description: str


@router.get("/api/characters/{character_id}/profile")
async def get_profile(character_id: str):
    character = _require_character(character_id)
    return character.profile.to_dict()


@router.put("/api/characters/{character_id}/profile")
async def update_profile(character_id: str, request: UpdateProfileRequest):
    character = _require_character(character_id)
    if request.biography is not None:
        character.profile.biography = request.biography
    if request.personality is not None:
        character.profile.personality = request.personality
    if request.physical is not None:
        from backend.models.profile import PhysicalTraits
        character.profile.physical = PhysicalTraits.from_dict(request.physical)
    if request.tendencies is not None:
        character.profile.tendencies = request.tendencies
    if request.expressions is not None:
        character.profile.expressions = request.expressions
    if request.notes is not None:
        character.profile.notes = request.notes
    character.emit("character_updated", character)
    return character.profile.to_dict()


@router.post("/api/characters/{character_id}/outfits")
async def add_outfit(character_id: str, request: AddOutfitRequest):
    character = _require_character(character_id)
    outfit = character.profile.add_outfit(request.name, request.description, request.is_default)
    character.emit("character_updated", character)
    return outfit.to_dict()


@router.post("/api/characters/{character_id}/relationships")
async def set_relationship(character_id: str, request: SetRelationshipRequest):
    character = _require_character(character_id)
    _require_character(request.target_character_id)
    character.profile.set_relationship(request.target_character_id, request.description)
    character.emit("character_updated", character)
    return character.profile.relationships


# --- Character chat ---

class ChatRequest(BaseModel):
    character_id: str
    message: str
    panel_id: Optional[str] = None
    history: Optional[list[dict]] = None


class SuggestScriptsRequest(BaseModel):
    character_id: str
    panel_id: str


@router.post("/api/chat/character")
async def chat_with_character(request: ChatRequest):
    """Chat with a character in-character. They respond based on their full profile + scene context."""
    character = _require_character(request.character_id)

    panel = None
    page = None
    if request.panel_id:
        panel = story.lookup_as(request.panel_id, Panel)
        if panel:
            context = panel.get_context()
            page_data = context.get("page", {})
            page_id = page_data.get("page_id")
            if page_id:
                from backend.models import Page as PageModel
                page = story.lookup_as(page_id, PageModel)

    from backend.generator.character_chat import CharacterChat
    chat = CharacterChat()
    response = await chat.chat(character, request.message, panel, page, request.history)

    return {
        "character_id": request.character_id,
        "character_name": character.name,
        "response": response,
    }


@router.post("/api/chat/react")
async def character_react(request: SuggestScriptsRequest):
    """Get a character's reaction to the current panel."""
    character = _require_character(request.character_id)
    panel = _require_panel(request.panel_id)

    page = None
    context = panel.get_context()
    page_id = context.get("page", {}).get("page_id")
    if page_id:
        from backend.models import Page as PageModel
        page = story.lookup_as(page_id, PageModel)

    from backend.generator.character_chat import CharacterChat
    chat = CharacterChat()
    reaction = await chat.react_to_panel(character, panel, page)
    return {"character_id": request.character_id, "reaction": reaction}


@router.post("/api/chat/suggest-scripts")
async def suggest_scripts(request: SuggestScriptsRequest):
    """LLM suggests script fields based on surrounding panels and character profile."""
    character = _require_character(request.character_id)
    panel = _require_panel(request.panel_id)

    # Find surrounding panels
    page = None
    previous_panel = None
    next_panel = None
    context = panel.get_context()
    page_id = context.get("page", {}).get("page_id")
    if page_id:
        from backend.models import Page as PageModel
        page = story.lookup_as(page_id, PageModel)
        if page:
            for i, p in enumerate(page.panels):
                if p.panel_id == panel.panel_id:
                    if i > 0:
                        previous_panel = page.panels[i - 1]
                    if i < len(page.panels) - 1:
                        next_panel = page.panels[i + 1]
                    break

    from backend.generator.character_chat import CharacterChat
    chat = CharacterChat()
    suggestions = await chat.suggest_scripts(character, panel, page, previous_panel, next_panel)
    return {"character_id": request.character_id, "suggestions": suggestions}


# --- Conversation bank ---

@router.post("/api/characters/{character_id}/conversations")
async def save_conversation(character_id: str, messages: list[dict]):
    """Save a conversation to the character's bank."""
    character = _require_character(character_id)
    character.conversations.append({
        "messages": messages,
        "saved_at": __import__("time").time(),
    })
    return {"count": len(character.conversations)}


@router.get("/api/characters/{character_id}/conversations")
async def list_conversations(character_id: str):
    character = _require_character(character_id)
    return character.conversations


# --- Chapters ---

@router.get("/api/chapters")
async def list_chapters():
    _require_story()
    return {
        chapter_id: chapter.to_dict()
        for chapter_id, chapter in story.chapters.items()
    }


@router.post("/api/chapters")
async def create_chapter(request: CreateChapterRequest):
    _require_story()
    # Validate all characters exist
    for character_id in request.character_ids:
        _require_character(character_id)

    if len(request.character_ids) == 0:
        raise HTTPException(400, "A chapter requires at least one character")

    chapter = story.create_chapter(
        title=request.title,
        character_ids=request.character_ids,
        synopsis=request.synopsis,
        default_location=request.default_location,
        default_time_of_day=request.default_time_of_day,
    )
    return chapter.to_dict()


@router.post("/api/chapters/{chapter_id}/characters/{character_id_to_add}")
async def add_character_to_chapter(chapter_id: str, character_id_to_add: str):
    """Add an existing character to a chapter. This extends the relationship."""
    chapter = _require_chapter(chapter_id)
    character = _require_character(character_id_to_add)

    if character_id_to_add in chapter.character_ids:
        return chapter.to_dict()  # already in chapter

    chapter.bind_character(character_id_to_add)
    character.add_chapter(chapter)
    return chapter.to_dict()


@router.put("/api/chapters/{chapter_id}")
async def update_chapter(chapter_id: str, request: UpdateChapterRequest):
    """Update chapter properties."""
    chapter = _require_chapter(chapter_id)
    if request.title is not None:
        chapter.title = request.title
    if request.synopsis is not None:
        chapter.synopsis = request.synopsis
    if request.default_location is not None:
        chapter.default_location = request.default_location
    if request.default_time_of_day is not None:
        chapter.default_time_of_day = request.default_time_of_day
    if request.negative_prompt is not None:
        chapter.negative_prompt = request.negative_prompt
    chapter.emit_up("chapter_updated", chapter)
    return chapter.to_dict()


# --- Pages ---

@router.post("/api/pages")
async def create_page(request: CreatePageRequest):
    """Create a page — cascades panel + scripts for chapter characters."""
    chapter = _require_chapter(request.chapter_id)

    page = chapter.create_page(layout_template=request.layout_template)
    story._register_cascade(chapter)
    return page.to_dict()


@router.put("/api/pages/{page_id}")
async def update_page(page_id: str, request: UpdatePageRequest):
    """Update page context — setting, mood, action, time, weather, lighting."""
    page = _require_page(page_id)
    if request.setting is not None:
        page.setting = request.setting
    if request.mood is not None:
        page.mood = request.mood
    if request.action_context is not None:
        page.action_context = request.action_context
    if request.time_of_day is not None:
        page.time_of_day = request.time_of_day
    if request.weather is not None:
        page.weather = request.weather
    if request.lighting is not None:
        page.lighting = request.lighting
    if request.negative_prompt is not None:
        page.negative_prompt = request.negative_prompt
    page.emit_up("page_updated", page)
    return page.to_dict()


# --- Panels ---

@router.post("/api/panels")
async def create_panel(request: CreatePanelRequest):
    """Create a panel — cascades scripts for chapter characters."""
    page = _require_page(request.page_id)

    # Find the chapter this page belongs to, to get character_ids
    context = page.get_context()
    character_ids = context.get("chapter", {}).get("character_ids", [])

    if not character_ids:
        raise HTTPException(400, "Cannot create panel — no characters in chapter")

    # Use the model's factory — it cascades scripts
    panel = page.create_panel(character_ids=character_ids)
    story.register_panel(panel)
    for script in panel.scripts.values():
        story.register_script(script)
    return panel.to_dict()


@router.put("/api/panels/{panel_id}")
async def update_panel(panel_id: str, request: UpdatePanelRequest):
    """Update panel properties — shot type, narration."""
    panel = _require_panel(panel_id)
    if request.shot_type is not None:
        panel.shot_type = request.shot_type
    if request.narration is not None:
        panel.narration = request.narration
    if request.negative_prompt is not None:
        panel.negative_prompt = request.negative_prompt
    panel.emit_up("panel_updated", panel)
    return panel.to_dict()


# --- Scripts ---

@router.post("/api/scripts")
async def create_script(request: CreateScriptRequest):
    panel = _require_panel(request.panel_id)
    character = _require_character(request.character_id)

    # Validate character is in this panel's chapter
    _require_character_in_hierarchy(request.character_id, panel)

    import uuid
    script_id = f"scr-{uuid.uuid4().hex[:8]}"
    script = Script(
        script_id=script_id,
        character_id=request.character_id,
        dialogue=request.dialogue,
        action=request.action,
        direction=request.direction,
        emotion=request.emotion,
        pose=request.pose,
        outfit=request.outfit,
    )
    script.source = "manual"
    panel.add_script(script)
    story.register_script(script)
    return script.to_dict()


@router.put("/api/scripts/{script_id}")
async def update_script(script_id: str, request: UpdateScriptRequest):
    script = story.lookup_as(script_id, Script)
    if not script:
        raise HTTPException(404, f"Script {script_id} not found")
    script.update(
        dialogue=request.dialogue,
        action=request.action,
        direction=request.direction,
        emotion=request.emotion,
        pose=request.pose,
        outfit=request.outfit,
        negative_prompt=request.negative_prompt,
        source=request.source,
    )
    return script.to_dict()


@router.delete("/api/scripts/{script_id}")
async def delete_script(script_id: str):
    """Remove a script from its panel. Cannot remove the last script."""
    script = story.lookup_as(script_id, Script)
    if not script:
        raise HTTPException(404, f"Script {script_id} not found")

    # Find the panel that owns this script
    for chapter in story.chapters.values():
        for page in chapter.pages:
            for panel in page.panels:
                if script.character_id in panel.scripts and panel.scripts[script.character_id].script_id == script_id:
                    panel.remove_script(script.character_id)
                    story.unregister(script_id)
                    return {"deleted": script_id}

    raise HTTPException(404, f"Script {script_id} not found in any panel")


# --- Generation ---

@router.post("/api/generate")
async def generate_panel_image(request: GeneratePanelRequest):
    """Generate an image for a panel. Requires the panel to have
    at least one script — no scripts means no characters means
    the hierarchy is incomplete.
    """
    panel = _require_panel(request.panel_id)

    if len(panel.scripts) == 0:
        raise HTTPException(
            400,
            "Panel has no scripts. Add at least one character script before generating. "
            "Hierarchy: Character -> Chapter -> Page -> Panel -> Script"
        )

    if not image_generator:
        raise HTTPException(503, "Image generator not loaded")

    if _generation_lock.locked():
        raise HTTPException(429, "Generation in progress, please wait")

    # Compose prompt from hierarchy context
    from backend.generator.panel_generator import PanelGenerator
    from backend.generator.ip_adapter_bridge import IPAdapterBridge
    characters = [
        story.get_character(character_id)
        for character_id in panel.scripts.keys()
        if story.get_character(character_id)
    ]

    # IP-Adapter conditioning from character reference banks
    ip_bridge = IPAdapterBridge(content_store) if content_store else None
    panel_gen = PanelGenerator(image_generator, ip_adapter_bridge=ip_bridge)
    prompt = await panel_gen.compose_prompt(panel, characters)
    # Compose negative from hierarchy (story + chapter + page + panel + characters)
    # User override takes priority if provided
    negative_prompt = request.negative_prompt or (
        panel_gen.prompt_composer.compose_negative(panel, characters)
    )

    generation_params = {
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
    }

    composer = panel_gen.prompt_composer
    print("=" * 60)
    print(f"GENERATING PANEL: {request.panel_id}")
    print(f"MODEL: {image_generator.model_id}")
    print(f"METHOD: {getattr(composer, 'last_method', 'unknown')}")
    print(f"PROMPT: {prompt}")
    print(f"NEGATIVE: {negative_prompt}")
    print(f"PARAMS: {generation_params}")
    print(f"CHARACTERS: {[c.name for c in characters]}")
    print("=" * 60)

    # Collect IP-Adapter reference images for conditioning
    ip_kwargs = {}
    if ip_bridge and image_generator.pipeline:
        ip_kwargs = ip_bridge.prepare_generation_kwargs(
            characters, panel, image_generator.pipeline,
            style_references=story.style_references if story else [],
        )

    async with _generation_lock:
        content_hash = await image_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **generation_params,
            **ip_kwargs,
        )

    panel.update_image(content_hash, source="ai")

    return {
        "content_hash": content_hash,
        "image_url": f"/api/content/{content_hash}",
        "prompt_used": prompt,
        "prompt_method": getattr(composer, "last_method", "unknown"),
        "prompt_direct": getattr(composer, "last_direct_prompt", None),
        "prompt_llm_input": getattr(composer, "last_llm_input", None),
        "prompt_llm_output": getattr(composer, "last_llm_prompt", None),
        "negative_prompt_used": negative_prompt,
        "parameters": generation_params,
        "characters_used": [
            {"name": c.name, "appearance": c.appearance_prompt}
            for c in characters
        ],
        "panel": panel.to_dict(),
    }


# --- Inpainting ---

@router.post("/api/inpaint")
async def inpaint_panel(request: InpaintRequest):
    """Inpaint a masked region of a panel's image.
    Requires the panel to have an existing image.
    """
    panel = _require_panel(request.panel_id)

    if not panel.image_hash:
        raise HTTPException(400, "Panel has no image to edit")

    if not image_generator:
        raise HTTPException(503, "Image generator not loaded")

    if _generation_lock.locked():
        raise HTTPException(429, "Generation in progress, please wait")

    print("=" * 60)
    print(f"INPAINTING PANEL: {request.panel_id}")
    print(f"PROMPT: {request.prompt}")
    print(f"SOURCE IMAGE: {panel.image_hash[:16]}...")
    print(f"STRENGTH: {request.strength}")
    print("=" * 60)

    from backend.generator.panel_generator import PanelGenerator
    negative_prompt = request.negative_prompt or PanelGenerator(image_generator).compose_negative_prompt()

    async with _generation_lock:
        content_hash = await image_generator.inpaint(
            image_hash=panel.image_hash,
            mask_base64=request.mask_data,
            prompt=request.prompt,
            negative_prompt=negative_prompt,
            steps=request.steps,
            strength=request.strength,
            seed=request.seed,
        )

    panel.update_image(content_hash, source="ai")

    return {
        "content_hash": content_hash,
        "image_url": f"/api/content/{content_hash}",
        "panel": panel.to_dict(),
    }


# --- Regenerate all ---


class RegenerateRequest(BaseModel):
    keep_good: bool = True


@router.post("/api/regenerate-all")
async def regenerate_all(request: RegenerateRequest):
    """Regenerate all panels in the current chapter.
    If keep_good=True, panels with positive feedback are skipped.
    Returns list of panels that need regeneration.
    """
    _require_story()

    # Find all panels across all chapters
    panels_to_regenerate = []
    panels_kept = []

    adapter = _get_adapter()
    positive_hashes = set()
    if adapter and hasattr(adapter, '_unified_trainer') and adapter._unified_trainer:
        positive_hashes = {
            p.visual_latent.data_ptr()
            for p in adapter._unified_trainer.pairs if p.accepted
        }

    # Check feedback for each panel
    good_image_hashes = set()
    if adapter:
        for entry in adapter.feedback:
            if entry.accepted:
                good_image_hashes.add(entry.content_hash)

    for chapter in story.chapters.values():
        for page in chapter.pages:
            for panel in page.panels:
                if not panel.scripts:
                    continue

                if (request.keep_good
                        and panel.image_hash
                        and panel.image_hash in good_image_hashes):
                    panels_kept.append({
                        "panel_id": panel.panel_id,
                        "reason": "rated good",
                    })
                else:
                    panels_to_regenerate.append({
                        "panel_id": panel.panel_id,
                        "has_image": panel.image_hash is not None,
                    })

    return {
        "to_regenerate": panels_to_regenerate,
        "kept": panels_kept,
        "total_panels": len(panels_to_regenerate) + len(panels_kept),
    }


# --- Image review (adversarial loop) ---


@router.post("/api/review/{panel_id}")
async def review_panel_image(panel_id: str):
    """Review a generated panel image — closes the adversarial loop.

    Captions the image using LLaVA, compares to the original prompt,
    returns the gap analysis.
    """
    panel = _require_panel(panel_id)
    if not panel.image_hash:
        raise HTTPException(400, "Panel has no image to review")
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = content_store.retrieve(panel.image_hash)
    if not image_bytes:
        raise HTTPException(404, "Image not found in content store")

    # Get the prompt that was used (stored in metadata)
    meta = content_store.get_meta(panel.image_hash)
    original_prompt = ""
    if meta and meta.metadata:
        original_prompt = meta.metadata.get("prompt", "")

    from backend.generator.image_reviewer import ImageReviewer
    reviewer = ImageReviewer()
    result = await reviewer.review(image_bytes, original_prompt)

    # Store review data on the panel for unified training
    panel._last_review = {
        "prompt": original_prompt,
        "caption": result.reverse_caption,
        "score": result.match_score,
    }

    # Also build the object graph context for this panel
    context = panel.get_context()
    character_context = context.get("character", {})
    chapter_context = context.get("chapter", {})
    page_context = context.get("page", {})
    object_context = (
        f"Character: {character_context.get('name', '')}. "
        f"{character_context.get('description', '')} "
        f"Appearance: {character_context.get('appearance_prompt', '')}. "
        f"Chapter: {chapter_context.get('synopsis', '')}. "
        f"Setting: {page_context.get('setting', '')}. "
        f"Mood: {page_context.get('mood', '')}."
    )
    panel._last_object_context = object_context

    return {
        "panel_id": panel_id,
        "original_prompt": result.original_prompt,
        "reverse_caption": result.reverse_caption,
        "match_score": result.match_score,
        "differences": result.differences,
        "suggestion": result.suggestion,
    }


@router.post("/api/caption")
async def caption_image_endpoint(content_hash: str):
    """Caption any image in the content store using LLaVA."""
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = content_store.retrieve(content_hash)
    if not image_bytes:
        raise HTTPException(404, "Image not found")

    from backend.generator.image_reviewer import ImageReviewer
    reviewer = ImageReviewer()
    caption = await reviewer.caption_image(image_bytes)

    return {"content_hash": content_hash, "caption": caption}


# --- Solo character generation ---


class SoloGenerateRequest(BaseModel):
    character_id: str
    prompt: str = ""
    pose: str = ""
    outfit: str = ""
    emotion: str = ""
    direction: str = ""
    shot_type: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    guidance_scale: float = 7.0
    seed: Optional[int] = None


@router.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image — extract character and art style descriptions.
    Returns structured fields that can auto-populate appearance and story settings.
    """
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = await file.read()

    # Store the image so it can be referenced later
    content_type = file.content_type or "image/png"
    content_hash = content_store.store(image_bytes, content_type)

    from backend.generator.image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer()
    result = await analyzer.analyze(image_bytes)

    return {
        "content_hash": content_hash,
        "image_url": f"/api/content/{content_hash}",
        "raw_caption": result.raw_caption,
        "character": {
            "species": result.character.species,
            "body_type": result.character.body_type,
            "height": result.character.height,
            "skin_tone": result.character.skin_tone,
            "hair_style": result.character.hair_style,
            "hair_colour": result.character.hair_colour,
            "eye_colour": result.character.eye_colour,
            "facial_features": result.character.facial_features,
            "outfit": result.character.outfit,
            "accessories": result.character.accessories,
            "pose": result.character.pose,
            "expression": result.character.expression,
        },
        "art_style": {
            "art_style": result.art_style.art_style,
            "colour_palette": result.art_style.colour_palette,
            "line_style": result.art_style.line_style,
            "rendering": result.art_style.rendering,
            "genre_hints": result.art_style.genre_hints,
        },
    }


@router.post("/api/characters/{character_id}/analyze-reference/{content_hash}")
async def analyze_reference(character_id: str, content_hash: str):
    """Analyze an existing reference image and return structured fields.
    Optionally auto-applies to the character's appearance properties.
    """
    character = _require_character(character_id)
    if not content_store:
        raise HTTPException(503, "Content store not available")

    image_bytes = content_store.retrieve(content_hash)
    if not image_bytes:
        raise HTTPException(404, "Image not found in content store")

    from backend.generator.image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer()
    result = await analyzer.analyze(image_bytes)

    # Auto-label the reference if it exists
    ref = character.appearance.get_reference(content_hash)
    if ref:
        if result.character.pose and not ref.pose:
            ref.pose = result.character.pose
        if result.character.expression and not ref.expression:
            ref.expression = result.character.expression
        if result.character.outfit and not ref.outfit_variant:
            ref.outfit_variant = result.character.outfit

    return {
        "content_hash": content_hash,
        "raw_caption": result.raw_caption,
        "character": {
            "species": result.character.species,
            "body_type": result.character.body_type,
            "height": result.character.height,
            "skin_tone": result.character.skin_tone,
            "hair_style": result.character.hair_style,
            "hair_colour": result.character.hair_colour,
            "eye_colour": result.character.eye_colour,
            "facial_features": result.character.facial_features,
            "outfit": result.character.outfit,
            "accessories": result.character.accessories,
            "pose": result.character.pose,
            "expression": result.character.expression,
        },
        "art_style": {
            "art_style": result.art_style.art_style,
            "colour_palette": result.art_style.colour_palette,
            "line_style": result.art_style.line_style,
            "rendering": result.art_style.rendering,
            "genre_hints": result.art_style.genre_hints,
        },
    }


@router.post("/api/characters/{character_id}/apply-analysis")
async def apply_analysis(character_id: str, request: dict):
    """Apply analyzed fields to a character's appearance properties.
    Also optionally sets story art_style.
    """
    _require_story()
    character = _require_character(character_id)

    char_fields = request.get("character", {})
    for field_name, value in char_fields.items():
        if value and hasattr(character.appearance.properties, field_name):
            setattr(character.appearance.properties, field_name, value)
    character.emit("character_updated", character)

    art_fields = request.get("art_style", {})
    if art_fields.get("art_style") and not story.art_style:
        story.art_style = art_fields["art_style"]
    if art_fields.get("genre_hints") and not story.genre:
        story.genre = art_fields["genre_hints"]

    return {
        "character": character.appearance.properties.to_dict(),
        "story_art_style": story.art_style,
        "story_genre": story.genre,
    }


@router.get("/api/characters/{character_id}/solo-chapter")
async def get_solo_chapter(character_id: str):
    """Get or create the solo chapter for a character."""
    _require_story()
    _require_character(character_id)
    solo = story.ensure_solo_chapter(character_id)
    return solo.to_dict()


@router.post("/api/generate-solo")
async def generate_solo(request: SoloGenerateRequest):
    """Generate a solo image of a character.
    Creates a panel in the character's solo chapter AND adds to reference bank.
    """
    _require_story()
    character = _require_character(request.character_id)

    if not image_generator:
        raise HTTPException(503, "Image generator not loaded")

    if _generation_lock.locked():
        raise HTTPException(429, "Generation in progress, please wait")

    # Compose prompt from all fields — like a proper panel
    prompt_parts = []
    if request.shot_type:
        prompt_parts.append(f"{request.shot_type} shot")
    if character.appearance_prompt:
        prompt_parts.append(character.appearance_prompt)
    if request.pose:
        prompt_parts.append(request.pose)
    if request.prompt:
        prompt_parts.append(request.prompt)
    if request.emotion:
        prompt_parts.append(f"({request.emotion})")
    if request.outfit:
        prompt_parts.append(f"wearing {request.outfit}")
    if request.direction:
        prompt_parts.append(f"[{request.direction}]")

    # Story art style
    if story.art_style:
        prompt_parts.insert(0, story.art_style)

    full_prompt = ", ".join(part for part in prompt_parts if part)

    from backend.generator.panel_generator import PanelGenerator
    negative = request.negative_prompt or (
        PanelGenerator(image_generator).compose_negative_prompt()
    )

    async with _generation_lock:
        content_hash = await image_generator.generate(
            prompt=full_prompt,
            negative_prompt=negative,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )

    # Add to reference bank with labels from the fields
    character.appearance.add_reference(
        content_hash=content_hash,
        source="generated",
        caption=request.prompt,
        pose=request.pose,
        expression=request.emotion,
        angle=request.direction,
        outfit_variant=request.outfit,
    )

    # Create a panel in the solo chapter (creates chapter if needed)
    solo = story.ensure_solo_chapter(character.character_id)
    if solo:
        # Add to the last page, or create a new one if full (4+ panels)
        last_page = solo.pages[-1] if solo.pages else None
        if not last_page or len(last_page.panels) >= 4:
            last_page = solo.create_page()
            story._register_cascade(solo)

        panel = last_page.create_panel(
            character_ids=[character.character_id]
        )
        panel.shot_type = request.shot_type
        # Set the script with all fields
        script = list(panel.scripts.values())[0]
        script.update(
            action=request.prompt,
            pose=request.pose,
            outfit=request.outfit,
            emotion=request.emotion,
            direction=request.direction,
            source="ai",
        )
        panel.update_image(content_hash, source="ai")
        story.register_panel(panel)
        story.register_script(script)

    character.emit("character_updated", character)

    return {
        "content_hash": content_hash,
        "image_url": f"/api/content/{content_hash}",
        "prompt": full_prompt,
        "negative_prompt": negative,
        "solo_chapter_id": solo.chapter_id if solo else None,
    }


# --- Feedback / Adapter training ---

# Per-story adapter
_story_adapters: dict = {}


def _get_adapter():
    """Get or create the adapter for the current story."""
    from backend.generator.adapter import StoryAdapter
    if not story:
        return None
    if story.story_id not in _story_adapters:
        _story_adapters[story.story_id] = StoryAdapter(story.story_id, content_store)
    return _story_adapters[story.story_id]


class FeedbackRequest(BaseModel):
    content_hash: str
    prompt: str
    accepted: bool
    character_ids: list[str] = []
    panel_id: str = ""


@router.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Thumbs up/down on a generated image. Feeds the adapter training."""
    adapter = _get_adapter()
    if not adapter:
        raise HTTPException(500, "No story loaded")

    entry = adapter.add_feedback(
        content_hash=request.content_hash,
        prompt=request.prompt,
        accepted=request.accepted,
        character_ids=request.character_ids,
        panel_id=request.panel_id,
    )

    # Save to character reference banks with accept/reject status
    for char_id in request.character_ids:
        character = story.get_character(char_id)
        if character:
            # Add if not already there
            if not character.appearance.get_reference(request.content_hash):
                character.appearance.add_reference(
                    content_hash=request.content_hash,
                    source="generated",
                    caption=request.prompt,
                )
            if request.accepted:
                character.appearance.accept_reference(request.content_hash)
            else:
                character.appearance.reject_reference(request.content_hash)

    # Capture training pair — includes review data if available
    if image_generator and image_generator._last_visual_latent is not None:
        if not hasattr(adapter, '_unified_trainer') or adapter._unified_trainer is None:
            from backend.generator.adversarial_adapter import AdversarialAdapter
            from backend.generator.unified_trainer import UnifiedTrainer
            hidden_dim = image_generator._last_visual_latent.shape[-1]
            adv_adapter = AdversarialAdapter(hidden_dim=hidden_dim, rank=adapter.lora_rank)
            adapter._unified_trainer = UnifiedTrainer(adv_adapter)
            # Keep legacy reference
            adapter._adversarial_trainer = type('', (), {
                'pair_count': lambda self: adapter._unified_trainer.pair_count(),
                'adapter': adv_adapter,
                'train': lambda self, **kw: [],
            })()

        if image_generator._last_language_latent is not None:
            vis = image_generator._last_visual_latent
            lang = image_generator._last_language_latent
            if vis.shape[-1] != lang.shape[-1]:
                min_dim = min(vis.shape[-1], lang.shape[-1])
                vis = vis[..., :min_dim]
                lang = lang[..., :min_dim]

            # Find the panel to get review data
            panel = story.lookup_as(request.panel_id, Panel) if request.panel_id else None
            review_caption = ""
            object_context = ""
            match_score = 0.5
            if panel:
                review = getattr(panel, '_last_review', {})
                review_caption = review.get("caption", "")
                match_score = review.get("score", 0.5)
                object_context = getattr(panel, '_last_object_context', "")

            adapter._unified_trainer.add_from_generation(
                visual_latent=vis,
                language_latent=lang,
                accepted=request.accepted,
                prompt_used=request.prompt,
                reverse_caption=review_caption,
                object_context=object_context,
                match_score=match_score,
            )

    return {
        "feedback": entry.to_dict(),
        "can_train": adapter.can_train(),
        "positive_count": len(adapter.positive_samples()),
        "negative_count": len(adapter.negative_samples()),
        "adversarial_pairs": (
            adapter._adversarial_trainer.pair_count()
            if getattr(adapter, '_adversarial_trainer', None) else 0
        ),
    }


@router.get("/api/feedback")
async def get_feedback():
    """Get all feedback for the current story."""
    adapter = _get_adapter()
    if not adapter:
        return {"feedback": [], "can_train": False}
    return {
        "feedback": [f.to_dict() for f in adapter.feedback],
        "can_train": adapter.can_train(),
        "positive_count": len(adapter.positive_samples()),
        "negative_count": len(adapter.negative_samples()),
        "adapter_hash": adapter.adapter_hash,
    }


class TrainRequest(BaseModel):
    rank: int = 4
    epochs: int = 100


@router.post("/api/adapter/train")
async def train_adapter(request: TrainRequest = TrainRequest()):
    """Train the story's LoRA adapter using collected feedback."""
    adapter = _get_adapter()
    if not adapter:
        raise HTTPException(500, "No story loaded")
    adapter.lora_rank = request.rank

    if not adapter.can_train():
        raise HTTPException(
            400,
            f"Not enough positive samples. Have {len(adapter.positive_samples())}, "
            f"need {adapter.min_training_samples}"
        )

    if not image_generator or not image_generator.pipeline:
        raise HTTPException(503, "Image generator not loaded")

    if _generation_lock.locked():
        raise HTTPException(429, "Generation in progress, cannot train now")

    # Auto-review all un-reviewed pairs before training
    if hasattr(adapter, '_unified_trainer') and adapter._unified_trainer:
        trainer = adapter._unified_trainer
        unreviewed = [
            p for p in trainer.pairs
            if p.image_embedding is None
        ]
        if unreviewed and content_store:
            from backend.generator.latent_reviewer import LatentReviewer
            reviewer = LatentReviewer()
            print(f"Auto-reviewing {len(unreviewed)} pairs...")
            for pair in unreviewed:
                # Find the panel to get image bytes and context
                panel = None
                if pair.prompt_used:
                    # Find by searching all panels for matching prompt
                    for ch in story.chapters.values():
                        for pg in ch.pages:
                            for pan in pg.panels:
                                if pan.image_hash:
                                    meta = content_store.get_meta(
                                        pan.image_hash
                                    )
                                    if (meta and meta.metadata and
                                            meta.metadata.get("prompt")
                                            == pair.prompt_used):
                                        panel = pan
                                        break

                if panel and panel.image_hash:
                    image_bytes = content_store.retrieve(panel.image_hash)
                    if image_bytes:
                        context = panel.get_context()
                        char_ctx = context.get("character", {})
                        object_text = (
                            f"{char_ctx.get('name', '')} "
                            f"{char_ctx.get('description', '')} "
                            f"{char_ctx.get('appearance_prompt', '')}"
                        )
                        review = await reviewer.review(
                            image_bytes, pair.prompt_used, object_text
                        )
                        if review:
                            pair.image_embedding = review.image_embedding
                            pair.prompt_embedding = review.prompt_embedding
                            pair.context_embedding = review.context_embedding
                            pair.reverse_caption = review.caption_text
                            pair.match_score = review.match_score

            reviewed_count = sum(
                1 for p in trainer.pairs if p.image_embedding is not None
            )
            print(f"Reviewed: {reviewed_count}/{len(trainer.pairs)} pairs")

    async with _generation_lock:
        adv_hash = None
        results = []

        # Unified training — all three loss components
        if hasattr(adapter, '_unified_trainer') and adapter._unified_trainer:
            trainer = adapter._unified_trainer
            if trainer.pair_count() > 0:
                results = trainer.train(epochs=request.epochs)
                adv_bytes = trainer.adapter.save_weights()
                adv_hash = content_store.store(
                    adv_bytes, "application/octet-stream", {
                        "type": "unified_adapter",
                        "story_id": story.story_id,
                        "pairs": trainer.pair_count(),
                        "reviewed": trainer.reviewed_pair_count(),
                    }
                )
                last = results[-1] if results else None
                if last:
                    print(
                        f"Unified adapter trained: "
                        f"vis={last.visual_loss:.4f} "
                        f"lang={last.language_loss:.4f} "
                        f"review={last.review_loss:.4f} "
                        f"align={last.alignment:.4f} "
                        f"pairs={trainer.pair_count()} "
                        f"reviewed={trainer.reviewed_pair_count()}"
                    )

                # Load trained weights into pipeline as LoRA
                if image_generator and image_generator.pipeline:
                    try:
                        from backend.generator.lora_bridge import LoraBridge
                        bridge = LoraBridge(trainer.adapter)
                        bridge.load_into_pipeline(image_generator.pipeline)
                        print("LoRA weights loaded into pipeline")
                    except Exception as e:
                        print(f"Failed to load LoRA weights: {e}")

    last_result = results[-1] if results else None
    return {
        "adapter_hash": adv_hash,
        "status": "trained" if adv_hash else "no_data",
        "pairs": (
            adapter._unified_trainer.pair_count()
            if getattr(adapter, '_unified_trainer', None) else 0
        ),
        "reviewed_pairs": (
            adapter._unified_trainer.reviewed_pair_count()
            if getattr(adapter, '_unified_trainer', None) else 0
        ),
        "visual_loss": last_result.visual_loss if last_result else None,
        "language_loss": last_result.language_loss if last_result else None,
        "review_loss": last_result.review_loss if last_result else None,
        "alignment": last_result.alignment if last_result else None,
    }


# --- Models ---

@router.get("/api/models")
async def list_models():
    """List available image generation models."""
    from backend.generator.image_generator import AVAILABLE_MODELS
    current = image_generator.model_id if image_generator else None
    return {
        "models": AVAILABLE_MODELS,
        "current": current,
    }


@router.post("/api/models/{model_key}")
async def switch_model(model_key: str):
    """Switch to a different image generation model. Downloads on first use."""
    from backend.generator.image_generator import AVAILABLE_MODELS
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, f"Model '{model_key}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    if not image_generator:
        raise HTTPException(503, "Image generator not initialized")

    if _generation_lock.locked():
        raise HTTPException(429, "Generation in progress, cannot switch models now")

    model_info = AVAILABLE_MODELS[model_key]
    async with _generation_lock:
        try:
            image_generator.load_model(model_info["id"])
        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {e}")

    return {
        "model_key": model_key,
        "model_id": model_info["id"],
        "name": model_info["name"],
    }


# --- Content serving ---

@router.get("/api/content/{content_hash}")
async def get_content(content_hash: str):
    if not content_store:
        raise HTTPException(503, "Content store not available")
    file_path = content_store.get_path(content_hash)
    if not file_path:
        raise HTTPException(404, "Content not found")
    meta = content_store.get_meta(content_hash)
    return FileResponse(file_path, media_type=meta.content_type)


# --- WebSocket ---

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"status": "connected"})
    except WebSocketDisconnect:
        pass


# --- Validation helpers ---

def _require_story():
    if not story:
        raise HTTPException(500, "No story loaded")


def _require_character(character_id: str) -> Character:
    _require_story()
    character = story.get_character(character_id)
    if not character:
        raise HTTPException(404, f"Character '{character_id}' not found")
    return character


def _require_chapter(chapter_id: str) -> Chapter:
    _require_story()
    chapter = story.get_chapter(chapter_id)
    if not chapter:
        raise HTTPException(404, f"Chapter '{chapter_id}' not found")
    return chapter


def _require_page(page_id: str) -> Page:
    _require_story()
    page = story.lookup_as(page_id, Page)
    if not page:
        raise HTTPException(404, f"Page '{page_id}' not found")
    return page


def _require_panel(panel_id: str) -> Panel:
    _require_story()
    panel = story.lookup_as(panel_id, Panel)
    if not panel:
        raise HTTPException(404, f"Panel '{panel_id}' not found")
    return panel


def _require_character_in_hierarchy(character_id: str, panel: Panel) -> None:
    """Validate a character belongs to the chapter that contains this panel."""
    context = panel.get_context()
    chapter_context = context.get("chapter", {})
    chapter_character_ids = chapter_context.get("character_ids", [])
    if character_id not in chapter_character_ids:
        raise HTTPException(
            400,
            f"Character '{character_id}' is not in this panel's chapter. "
            f"Chapter characters: {chapter_character_ids}"
        )
