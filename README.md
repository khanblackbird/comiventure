# Comiventure

An interactive comic book adventure game powered by local AI models. Players chat with characters in a visual novel / dating sim style interface, and the AI generates both the narrative and comic book panels in real-time.

## Concept

- Players interact with characters through dialogue choices and free-text chat
- An LLM drives character personalities, story branching, and narrative progression
- An image generation model creates comic panels for each scene
- Panels can be animated using video generation models
- Players can edit generated content — mask a region and prompt the AI to change it (e.g. swap a character, change colours, alter the scene)
- Pages are laid out in comic book format with panels, speech bubbles, and narration boxes
- Entirely local — no API calls, no cloud dependencies

## Architecture

**Strictly OO Python backend** serving a **web frontend** via FastAPI.

```
+-------------------------------------------------------------------+
|                        Web Frontend                                |
|  HTML5 Canvas/WebGL + CSS Grid + vanilla JS                       |
|                                                                    |
|  - Comic page layout (CSS Grid panels)                            |
|  - Speech bubbles and narration boxes                              |
|  - Chat input / dialogue choices                                   |
|  - Animated panels (HTML5 video in-line)                           |
|  - Mask drawing tool (canvas) for AI edits                         |
|  - Region select + colour/style transforms (WebGL shaders)         |
|  - Page history, save/load                                         |
+-------------------------------------------------------------------+
        |  WebSocket / HTTP
        v
+-------------------------------------------------------------------+
|                     Python Backend (FastAPI)                        |
|                                                                    |
|  +------------------+    +-------------------+                     |
|  |   StoryEngine    |    |   EditEngine      |                     |
|  |  - CharacterAI   |    |  - Inpainting     |  mask + prompt      |
|  |  - SceneManager  |    |    (image/video)   |  -> edited asset   |
|  |  - DialogueTree  |    |  - ColourTransform |                     |
|  |  - BranchManager |    |  - RegionSwap      |                     |
|  +------------------+    +-------------------+                     |
|           |                       |                                |
|           v                       v                                |
|  +------------------+    +-------------------+                     |
|  |  PanelGenerator  |    |   ComfyUI Bridge  |  <- shared         |
|  |  - SceneRender   |    |  - ImageGen       |                     |
|  |  - CharacterGen  |    |  - VideoGen       |                     |
|  |  - BackgroundGen |    |  - Inpaint        |                     |
|  +------------------+    +-------------------+                     |
|           |                                                        |
|           v                                                        |
|  +------------------+                                              |
|  |  ComicComposer   |                                              |
|  |  - PageLayout    |                                              |
|  |  - BubblePlacer  |                                              |
|  |  - PanelArranger |                                              |
|  +------------------+                                              |
+-------------------------------------------------------------------+
        |                       |
        v                       v
+----------------+    +-------------------+
|    ollama      |    |     ComfyUI       |
|  (LLM server)  |    |  (image/video)    |
|  localhost:11434|    |  localhost:8188   |
+----------------+    +-------------------+
```

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Backend | Python + FastAPI | OO architecture, serves API to frontend |
| LLM | ollama (Llama 3 8B or similar) | Character dialogue, story generation, scene descriptions |
| Image Gen | ComfyUI + Flux/SDXL | Panel art, character portraits, backgrounds |
| Image Editing | ComfyUI inpainting | Mask + prompt to edit regions of generated images |
| Video/Animation | AnimateDiff (via ComfyUI) | Animated panels, motion from static images |
| Video Editing | ProPainter / AnimateDiff inpainting | Mask + prompt to edit regions across video frames |
| Frontend | HTML5 Canvas/WebGL + CSS Grid | Comic layout, mask drawing, animated panels, shader effects |
| Communication | WebSocket + REST | Real-time updates for generation progress |

## AI Editing Capabilities

Players can interactively edit generated content:

- **Inpainting** — draw a mask over a region, describe what to change ("make the cat a dog", "change shirt to orange"), AI regenerates just that area
- **Image inpainting** — single panel edits via Flux/SDXL inpainting pipeline
- **Video inpainting** — edit regions across animated panel frames with temporal consistency
- **Style transfer** — apply colour/style changes to selected regions via WebGL shaders (instant) or AI re-generation (higher quality)

Workflow: user draws mask in browser canvas -> frontend sends mask + prompt to backend -> ComfyUI processes the edit -> backend returns updated asset -> frontend displays it.

## Hardware Requirements

Developed on:
- NVIDIA RTX 4060 Mobile (8GB VRAM)
- Intel i7-13620H
- 16GB RAM
- Arch Linux

Minimum: Any NVIDIA GPU with 8GB VRAM and CUDA support.

8GB VRAM supports:
- LLM inference (7-8B parameter models)
- Image generation (Flux/SDXL)
- Image inpainting
- Short video clips (AnimateDiff, 2-4 sec)

Larger VRAM (12-24GB) unlocks:
- Bigger LLMs (13B+) for richer dialogue
- Higher resolution video generation
- Faster batch generation of multiple panels

## Design Principles

- **Fully offline** — all models run locally, no API keys needed
- **Strictly OO** — clean class hierarchies, domain objects, separation of concerns
- **Model-agnostic** — swap in bigger/better models later without code changes
- **Modular** — each component (story, image, edit, layout, UI) is independent
- **Lightweight first** — start with small models, scale up when hardware allows
- **Web frontend** — no game engine overhead, HTML/CSS/Canvas handles layout, animation, and editing

## Project Structure

```
comiventure/
  backend/
    engine/          # StoryEngine, CharacterAI, SceneManager, BranchManager
    generator/       # PanelGenerator, ComfyUI bridge, image/video pipelines
    editor/          # EditEngine, inpainting, region transforms
    composer/        # ComicComposer, PageLayout, BubblePlacer
    models/          # Domain objects: Character, Scene, Panel, Page, Story
    api/             # FastAPI routes and WebSocket handlers
    app.py           # Application entry point
  frontend/
    index.html       # Main page
    css/             # Comic layout styles, panel grids, bubble styles
    js/              # Canvas mask tool, WebGL shaders, chat UI, panel viewer
    assets/          # Static assets (fonts, UI elements)
  tests/
  requirements.txt
  README.md
```

## Status

Early development. Setting up the local AI stack.

## Getting Started

### 1. Install ollama
```bash
sudo pacman -S ollama
```

### 2. Pull a model
```bash
ollama pull llama3:8b
```

### 3. Install ComfyUI
TBD

### 4. Run Comiventure
TBD
