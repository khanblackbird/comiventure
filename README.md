# Comiventure

Interactive comic book adventure engine powered by local AI models. Create characters, compose scenes, and generate comic panels — all running on your own hardware.

## What It Does

- **Comic creation**: Story → Character → Chapter → Page → Panel → Script hierarchy
- **AI image generation**: SDXL models (5 built-in + custom checkpoint upload from Civitai etc.)
- **LoRA support**: Upload LoRAs from HuggingFace/Civitai, or train your own from feedback
- **Character chat**: Talk to characters in-character via Llama 3 8B
- **Adversarial training**: User feedback (thumbs up/down) trains a LoRA adapter for style consistency
- **IP-Adapter**: Character reference images condition generation for visual consistency
- **Image analysis**: Upload character art, LLaVA extracts appearance/art style automatically
- **Review loop**: LLaVA reverse-captions generated images, compares to prompts, suggests improvements
- **Save/load**: Stories saved as .cvn files (ZIP archives with JSON + assets)

## Architecture

```
Story (art_style, genre)
  └─ Character (appearance, profile, reference bank)
       └─ Chapter (location, time_of_day, synopsis)
            └─ Page (setting, mood, weather, lighting, action_context)
                 └─ Panel (shot_type, narration, image_hash)
                      └─ Script (dialogue, action, emotion, pose, outfit, direction)
```

Every object has a standard `to_prompt()` method. The prompt composer chains them:

```
Story.to_prompt()      → "cinematic lighting, hyper-detailed textures"
Panel.to_prompt()      → "close-up shot"
Character.to_prompt()  → "blue-haired girl, slim, blue eyes"
  + Script.to_prompt() → "standing, waves (wary) wearing armor [close-up]"
Page.to_prompt()       → "enchanted forest, dusk, rain, moonlight lighting, tense atmosphere"
```

### AI Pipeline

```
User edits scripts/context
        ↓
PromptComposer (LLM or direct fallback)
        ↓
ImageGenerator (SDXL + CPU offload)
  ├── Latent capture (visual + language)
  ├── IP-Adapter conditioning (reference images)
  └── LoRA weights (from adversarial adapter)
        ↓
ContentStore (SHA-256 content-addressable)
        ↓
User feedback (thumbs up/down)
        ↓
AdversarialAdapter training (visual + language + review losses)
        ↓
LoraBridge → load_lora_weights() into pipeline
```

### Backend

| Module | Purpose |
|--------|---------|
| `backend/models/` | Domain objects with emission architecture, dirty-flag caching, context inheritance. Objects hold state and emit events — no work logic. |
| `backend/generator/` | Image generation, prompt composition, adversarial adapter, LoRA bridge, IP-Adapter, image analysis. All external calls use logging, not print. |
| `backend/api/routes.py` | FastAPI REST endpoints for CRUD, generation, feedback, training, model switching. File uploads are path-sanitized. |
| `backend/composer/` | Comic page layout computation (CSS grid templates) |
| `backend/models/storage.py` | Save/load .cvn files (ZIP with JSON + content-addressed assets) |

### Frontend

| File | Purpose |
|------|---------|
| `frontend/index.html` | All screens: splash, chapter select, character screen, comic editor |
| `frontend/js/app.js` | Main application logic, screen navigation, UI wiring |
| `frontend/js/api.js` | API client |
| `frontend/js/comic.js` | Comic panel renderer (CSS grid, speech bubbles, selection) |
| `frontend/js/editor.js` | Inline mask editor for inpainting |

## Getting Started

### Using start.sh (recommended)

```bash
./start.sh
```

The startup script runs preflight checks (Docker, GPU, disk space, swap, Ollama, CUDA containers, image rebuild), auto-pulls from git if clean, then starts the server with a health check. See all commands:

```bash
./start.sh              # Start server
./start.sh proxy        # Start with Civitai WARP proxy
./start.sh restart      # Restart app container
./start.sh stop         # Stop everything
./start.sh logs         # Tail app logs
./start.sh test         # Run backend tests
./start.sh test-ui      # Run Playwright UI tests
./start.sh test-all     # Run all tests
./start.sh setup-proxy  # First-time WARP proxy setup
```

### Docker (manual)

```bash
docker compose up --build
```

Requires NVIDIA GPU with Docker GPU support. The compose file runs the app + Ollama containers sharing the GPU.

### Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama (for LLM + LLaVA)
ollama pull llama3:8b
ollama pull llava:7b

uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

### First Use

1. Click "New Story" on the splash screen
2. Go to Characters → "New from Image" to upload character art
3. LLaVA analyzes the image and auto-fills appearance + art style
4. Create chapters, add pages, fill in scripts
5. Generate panels — the AI composes prompts from your hierarchy

## Image Models

### Built-in Models

| Key | Model | Style |
|-----|-------|-------|
| `anime` | Lykon/AAM_XL_AnimeMix | Clean anime illustration |
| `pony` | CitronLegacy/ponyDiffusionV6XL | Anime + furry/anthro (Danbooru, e621) |
| `animagine` | cagliostrolab/animagine-xl-3.1 | High quality anime |
| `furry` | John6666/nova-furry-xl-il-v120-sdxl | Furry/anthro specialist |
| `autismmix` | John6666/autismmix-sdxl-autismmix-pony-sdxl | Anime + furry blend |

Models download on first use. Switch from the chapter select screen.

### Custom Checkpoints

Upload `.safetensors` checkpoint files (from Civitai etc.) via the "Upload Checkpoint" button on the chapter select screen. Any SDXL-compatible checkpoint works. Uploaded checkpoints appear in the model selector as `local:<name>`.

### LoRAs

Upload LoRA files or browse HuggingFace/Civitai from the Style LoRAs section. LoRAs stack on top of the active base model with adjustable strength.

## Training

The adversarial adapter learns from user feedback:

1. Generate a panel → rate it (thumbs up/down) — one vote per image
2. LLaVA reviews it → captures latent embeddings
3. Train adapter (configurable rank + epochs)
4. Trained weights convert to LoRA via LoraBridge and load into the pipeline
5. Next generation uses the adapted model

Training parameters (rank, epochs) are adjustable from the chapter select screen. The feedback counter appears next to the Train button.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `HF_TOKEN` | (none) | HuggingFace token for gated models |
| `CIVITAI_PROXY` | (none) | SOCKS5 proxy for Civitai downloads |

## Hardware

Developed on NVIDIA RTX 4060 Mobile (8GB VRAM, 15GB RAM). Uses sequential CPU offload for SDXL models.

- **Minimum**: 8GB VRAM NVIDIA GPU, 16GB system RAM (or swap)
- **Recommended**: 12GB+ VRAM for faster generation

The startup script auto-creates a 16GB swap file if none exists.

## Troubleshooting

**CUDA not detected in container**: Usually a nvidia-container-toolkit / driver version mismatch. `start.sh` auto-detects this and adds `privileged: true` to docker-compose.yml as a workaround.

**OOM during generation**: SDXL with CPU offload needs system RAM + swap. Run `./start.sh` — it auto-creates swap if missing.

**Image analysis fails**: Check that Ollama is running and has `llava:7b` pulled. The analysis timeout is 180 seconds for large images.

**503 Service Unavailable on model switch**: The image generator initializes on first use. If CUDA isn't available, all generation endpoints return 503.

**Story won't save (integrity violation)**: The save process auto-repairs orphaned scripts and empty chapters before validation. If it still fails, there may be a deeper data issue — check the server logs.

## Testing

```bash
./start.sh test          # Backend tests (fast, no GPU needed)
./start.sh test-ui       # UI tests (Playwright, needs running server)
./start.sh test-all      # Both
```

Test coverage:
- Model hierarchy (creation, traversal, context, validation, repair)
- Character removal cascading (scripts, solo chapters, bindings)
- Feedback deduplication (one vote per image)
- Content store (SHA-256, CRUD, round-trip)
- Storage (save/load .cvn with all fields)
- Image generation (prompt composition, latent capture, pipeline integrity)
- Adversarial training (adapter, unified trainer, LoRA bridge dimension projection)
- IP-Adapter (reference collection, pipeline loading, failure handling)
- Appearance and profile (structured properties, reference bank)
- Prompt composition (to_prompt chain, character filtering, negative hierarchy)
- API hierarchy enforcement and integrity
- End-to-end workflows

## Design Principles

- **Emission architecture**: All domain objects inherit from `Emitter`. Objects hold state and emit events upward — they don't call each other's methods. Events propagate via `emit_up()` through the parent chain. Context inherits downward via `get_context()` with dirty-flag caching.
- **Content-addressable storage**: Only SHA-256 hashes travel through emission — never pixels. The `ContentStore` handles all binary data separately.
- **Flat registry**: The Story maintains an O(1) lookup of every object by ID. No tree traversal needed.
- **Hierarchy validation**: `validate()` walks the entire graph checking integrity. `repair()` auto-fixes common issues (orphaned scripts, dangling references).
- **Logging over print**: All output goes through Python `logging` module for configurable verbosity.

## File Format

Stories are saved as `.cvn` files — ZIP archives containing:
- `story.json` — full hierarchy serialized
- `content/` — all images/assets by SHA-256 hash
