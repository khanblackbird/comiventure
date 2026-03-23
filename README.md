# Comiventure

An interactive comic book adventure game powered by local AI models. Players chat with characters in a visual novel / dating sim style interface, and the AI generates both the narrative and comic book panels in real-time.

## Concept

- Players interact with characters through dialogue choices and free-text chat
- An LLM drives character personalities, story branching, and narrative progression
- An image generation model creates comic panels for each scene
- Pages are laid out in comic book format with panels, speech bubbles, and narration boxes
- Entirely local — no API calls, no cloud dependencies

## Architecture

```
Player Input (chat/choices)
        |
        v
+------------------+
|   Story Engine   |  <- LLM (ollama, 8B model)
|  - Character AI  |     Manages dialogue, personality, story state
|  - Scene logic   |     Generates scene descriptions for image gen
|  - Branching     |
+------------------+
        |
        v
+------------------+
|  Panel Generator |  <- Image gen (ComfyUI + Flux/SDXL)
|  - Scene render  |     Generates character art and backgrounds
|  - Consistent    |     Maintains visual consistency across panels
|    characters    |
+------------------+
        |
        v
+------------------+
|  Comic Composer  |  <- Layout engine
|  - Panel layout  |     Arranges panels on pages
|  - Speech bubbles|     Adds dialogue bubbles and narration
|  - Page assembly |     Outputs finished comic pages
+------------------+
        |
        v
+------------------+
|    Frontend UI   |  <- Player-facing interface
|  - Read panels   |     Displays comic pages
|  - Chat input    |     Dialogue input and choice selection
|  - Story nav     |     Page history, save/load
+------------------+
```

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| LLM | ollama (Llama 3 8B or similar) | Character dialogue, story generation, scene descriptions |
| Image Gen | ComfyUI + Flux/SDXL | Panel art, character portraits, backgrounds |
| Video/Animation | AnimateDiff (future) | Optional animated panels |
| Frontend | TBD | Web UI or native app |
| Language | Python | Orchestration and backend |

## Hardware Requirements

Developed on:
- NVIDIA RTX 4060 Mobile (8GB VRAM)
- Intel i7-13620H
- 16GB RAM
- Arch Linux

Minimum: Any NVIDIA GPU with 8GB VRAM and CUDA support.

## Design Principles

- **Fully offline** — all models run locally, no API keys needed
- **Model-agnostic** — swap in bigger/better models later without code changes
- **Modular** — each component (story, image, layout, UI) is independent
- **Lightweight first** — start with small models, scale up when hardware allows

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
