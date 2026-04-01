# Comiventure

Interactive comic book adventure engine powered by local AI models. Create characters, compose scenes, and generate comic panels — all running on your own hardware.

## What It Does

- **Comic creation**: Story → Character → Chapter → Page → Panel → Script hierarchy
- **AI image generation**: SDXL models (5 built-in: anime, pony, furry, animagine, autismmix)
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
| `backend/models/` | Domain objects (Story, Character, Chapter, Page, Panel, Script) with emission, dirty-flag caching, context inheritance |
| `backend/generator/` | Image generation (SDXL), prompt composition (LLM), adversarial adapter, LoRA bridge, IP-Adapter, image analysis, review |
| `backend/api/routes.py` | FastAPI REST endpoints for all CRUD, generation, feedback, training, model switching |
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

### Screens

1. **Splash** → New Story / Load File / Saved Stories
2. **Chapter Select** → Story settings (art style, genre), chapter cards (location, time, synopsis), model selector, adapter training
3. **Character Screen** → Character list, appearance editor, solo chapter panels, reference bank, upload & analyze
4. **Comic Editor** → Page viewer, panel selection, script editing, generation, feedback, review, chat

## Getting Started

### Docker (recommended)

```bash
docker compose up --build
```

Requires NVIDIA GPU with Docker GPU support. The compose file runs the app + ollama containers sharing the GPU.

### Manual Setup

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama (for LLM + LLaVA)
# Install ollama, then:
ollama pull llama3:8b
ollama pull llava:7b

# Run
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### First Use

1. Click "New Story" on the splash screen
2. Go to Characters → "New from Image" to upload character art
3. LLaVA analyzes the image and auto-fills appearance + art style
4. Create chapters, add pages, fill in scripts
5. Generate panels — the AI composes prompts from your hierarchy

## Image Models

| Key | Model | Style |
|-----|-------|-------|
| `anime` | Lykon/AAM_XL_AnimeMix | Clean anime illustration |
| `pony` | CitronLegacy/ponyDiffusionV6XL | Anime + furry/anthro (Danbooru, e621) |
| `animagine` | cagliostrolab/animagine-xl-3.1 | High quality anime |
| `furry` | John6666/nova-furry-xl-il-v120-sdxl | Furry/anthro specialist |
| `autismmix` | John6666/autismmix-sdxl-autismmix-pony-sdxl | Anime + furry blend |

Models download on first use. Switch models from the chapter select screen.

## Default Prompts

**Style** (prepended to all prompts): `cinematic lighting, hyper-detailed textures`

**Negative** (default for all generation):
```
lowres, (worst quality, bad quality:1.2), bad anatomy, sketch, jpeg artefacts,
signature, watermark, old, oldest, censored, bar_censor, simple background
```

Both are overridable — art style via Story settings, negative prompt per panel.

## Training

The adversarial adapter learns from user feedback:

1. Generate a panel → rate it (thumbs up/down)
2. LLaVA reviews it → captures latent embeddings
3. Train adapter (configurable rank + epochs)
4. Trained weights convert to LoRA via LoraBridge
5. Next generation uses the adapted model

Training parameters (rank, epochs) are adjustable from the chapter select screen.

## Hardware

Developed on NVIDIA RTX 4060 Mobile (8GB VRAM). Uses sequential CPU offload for SDXL models.

- **Minimum**: 8GB VRAM NVIDIA GPU with CUDA
- **Recommended**: 12GB+ VRAM for faster generation

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

Test coverage:
- Model hierarchy (creation, traversal, context, validation)
- Content store (SHA-256, CRUD, round-trip)
- Storage (save/load .cvn with all fields)
- Image generation (prompt composition, latent capture, pipeline integrity)
- Adversarial training (adapter, unified trainer, LoRA bridge)
- IP-Adapter (reference collection, pipeline loading)
- Appearance and profile (structured properties, reference bank)
- Prompt composition (to_prompt chain, character filtering, defaults)
- API hierarchy enforcement and integrity
- End-to-end workflows

## File Format

Stories are saved as `.cvn` files — ZIP archives containing:
- `story.json` — full hierarchy serialized
- `content/` — all images/assets by SHA-256 hash
