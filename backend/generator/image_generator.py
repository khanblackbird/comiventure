"""Image generation using a frozen Pony Diffusion base with diffusers.

The generator is an Emitter — it plugs into the hierarchy through events,
not function calls. It receives context from the hierarchy and emits
results back. This means it's swappable: replace this with ComfyUI,
a different model, or manual input without touching anything else.

Architecture:
  Frozen base (Pony Diffusion) → knows how to draw anime/furry
  Unfrozen adapter (future)    → learns this story's style via user feedback
  Hierarchy context             → character appearances, chapter mood, scripts
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from backend.models.emitter import Emitter
from backend.models.content_store import ContentStore


AVAILABLE_MODELS = {
    "anime": {
        "id": "Lykon/AAM_XL_AnimeMix",
        "name": "AAM XL AnimeMix",
        "description": "Anime style — clean, detailed illustration",
        "tags": ["anime", "manga", "illustration"],
    },
    "pony": {
        "id": "CitronLegacy/ponyDiffusionV6XL_Diffusers",
        "name": "Pony Diffusion V6 XL",
        "description": "Anime + furry/anthro — trained on Danbooru, e621, Derpibooru",
        "tags": ["anime", "furry", "anthro", "nsfw"],
    },
    "animagine": {
        "id": "cagliostrolab/animagine-xl-3.1",
        "name": "Animagine XL 3.1",
        "description": "High quality anime — most popular anime SDXL model",
        "tags": ["anime", "high-quality"],
    },
    "furry": {
        "id": "John6666/nova-furry-xl-il-v120-sdxl",
        "name": "Nova Furry XL",
        "description": "Furry/anthro specialist — e621 trained",
        "tags": ["furry", "anthro", "nsfw"],
    },
    "autismmix": {
        "id": "John6666/autismmix-sdxl-autismmix-pony-sdxl",
        "name": "AutismMix Pony SDXL",
        "description": "Anime + furry blend — versatile style",
        "tags": ["anime", "furry", "versatile"],
    },
}

DEFAULT_MODEL = "anime"


class ImageGenerator(Emitter):
    """Generates images from text prompts using a frozen diffusion model.

    Supports multiple models — switch at runtime without restarting.
    Emits 'generation_started', 'generation_progress', 'generation_complete'.
    Output goes into the ContentStore — only hashes propagate.
    """

    def __init__(
        self,
        content_store: ContentStore,
        model_id: str = None,
        device: str = "cuda",
        output_dir: str = "data/generated",
    ) -> None:
        super().__init__()
        self.content_store = content_store
        self.model_id = model_id or AVAILABLE_MODELS[DEFAULT_MODEL]["id"]
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = None
        self._inpaint_pipeline = None
        self._loaded = False
        self._loaded_model_id = None
        self.adversarial = None  # AdversarialAdapter, set per-story
        self._last_visual_latent = None
        self._last_language_latent = None

    def load_model(self, model_id: str = None) -> None:
        """Load a model. Pass a different model_id to switch models."""
        if model_id:
            self.model_id = model_id

        # Skip if this exact model is already loaded
        if self._loaded and self._loaded_model_id == self.model_id:
            return

        from diffusers import StableDiffusionXLPipeline
        import torch

        log.info("Loading model: %s", self.model_id)

        local_path = Path(self.model_id)
        if local_path.exists() and local_path.suffix == ".safetensors":
            # Local single-file checkpoint (e.g. from Civitai)
            log.info("Loading local checkpoint: %s", local_path.name)
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                str(local_path),
                torch_dtype=torch.float16,
            )
        else:
            # HuggingFace model ID — try local cache first
            try:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                )
                log.info("Loaded from cache: %s", self.model_id)
            except Exception:
                # Fall back to downloading
                log.info("Not cached, downloading: %s", self.model_id)
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                )

        # Sequential CPU offload — required for 8GB VRAM with SDXL
        self.pipeline.enable_sequential_cpu_offload()

        # Freeze all base model parameters
        for parameter in self.pipeline.unet.parameters():
            parameter.requires_grad = False
        for parameter in self.pipeline.vae.parameters():
            parameter.requires_grad = False
        for parameter in self.pipeline.text_encoder.parameters():
            parameter.requires_grad = False

        # Memory optimisations
        self.pipeline.enable_attention_slicing()
        self.pipeline.vae.enable_tiling()

        # Unload any existing LoRA before switching models
        if self._loaded and self.pipeline is not None:
            try:
                self.pipeline.unload_lora_weights()
            except Exception:
                pass  # no LoRA loaded — fine

        # Reset inpaint pipeline so it rebuilds from new base
        self._inpaint_pipeline = None

        self._loaded = True
        self._loaded_model_id = self.model_id
        self.emit("model_loaded", {"model_id": self.model_id})

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, low quality, blurry, deformed",
        width: int = 768,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        ip_adapter_image: Optional[list] = None,
    ) -> str:
        """Generate an image and store it. Returns the content hash.

        The hash is what travels through the hierarchy — not pixels.
        ip_adapter_image: optional list of PIL images for IP-Adapter conditioning.
        """
        if not self._loaded:
            self.load_model()

        self.emit("generation_started", {
            "prompt": prompt,
            "width": width,
            "height": height,
        })

        # Run inference in a thread to keep async working
        result = await asyncio.to_thread(
            self._run_inference,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            ip_adapter_image=ip_adapter_image,
        )

        # Convert to PNG bytes and store
        from io import BytesIO
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        content_hash = self.content_store.store(
            image_bytes,
            "image/png",
            metadata={
                "width": width,
                "height": height,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "model_id": self.model_id,
            },
        )

        self.emit("generation_complete", {
            "content_hash": content_hash,
            "prompt": prompt,
        })

        return content_hash

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: Optional[int] = None,
        ip_adapter_image: Optional[list] = None,
    ):
        """Synchronous inference — called in a thread.
        Captures latents for adversarial training without extra overhead.
        """
        import torch

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Capture latents during generation via callback — no extra inference.
        # The pipeline already computes prompt embeds and UNet latents internally.
        captured = {"visual": None, "language": None}

        def capture_callback(pipe, step_index, timestep, callback_kwargs):
            if step_index == steps - 1:
                latents = callback_kwargs.get("latents", None)
                if latents is not None:
                    captured["visual"] = latents.detach().cpu()

                prompt_embeds = callback_kwargs.get("prompt_embeds", None)
                if prompt_embeds is not None:
                    captured["language"] = prompt_embeds.detach().cpu()
            return callback_kwargs

        # Build pipeline kwargs — IP-Adapter images added if present
        pipeline_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=capture_callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        )

        if ip_adapter_image:
            pipeline_kwargs["ip_adapter_image"] = ip_adapter_image

        try:
            output = self.pipeline(**pipeline_kwargs)
        except TypeError:
            # Fallback if pipeline doesn't support tensor input capture
            fallback_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            if ip_adapter_image:
                fallback_kwargs["ip_adapter_image"] = ip_adapter_image
            output = self.pipeline(**fallback_kwargs)

        if captured["visual"] is not None:
            self._last_visual_latent = captured["visual"].mean(dim=(2, 3))
        if captured["language"] is not None:
            self._last_language_latent = captured["language"].mean(dim=1)

        return output.images[0]

    async def inpaint(
        self,
        image_hash: str,
        mask_base64: str,
        prompt: str,
        negative_prompt: str = "worst quality, low quality, blurry, deformed",
        steps: int = 25,
        guidance_scale: float = 7.0,
        strength: float = 0.75,
        seed: Optional[int] = None,
    ) -> str:
        """Inpaint a masked region of an existing image. Returns the new content hash."""
        if not self._loaded:
            self.load_model()

        self.emit("generation_started", {"prompt": prompt, "type": "inpaint"})

        result = await asyncio.to_thread(
            self._run_inpaint,
            image_hash=image_hash,
            mask_base64=mask_base64,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

        from io import BytesIO
        buffer = BytesIO()
        result.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        content_hash = self.content_store.store(
            image_bytes,
            "image/png",
            metadata={
                "prompt": prompt,
                "type": "inpaint",
                "source_image": image_hash,
                "model_id": self.model_id,
            },
        )

        self.emit("generation_complete", {"content_hash": content_hash, "type": "inpaint"})
        return content_hash

    def _run_inpaint(
        self,
        image_hash: str,
        mask_base64: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None = None,
    ):
        """Synchronous inpainting — called in a thread."""
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        from diffusers import AutoPipelineForInpainting

        # Load source image from content store
        source_bytes = self.content_store.retrieve(image_hash)
        if not source_bytes:
            raise ValueError(f"Source image {image_hash} not found in content store")
        source_image = Image.open(BytesIO(source_bytes)).convert("RGB")

        # Decode mask from base64
        mask_bytes = base64.b64decode(mask_base64)
        mask_image = Image.open(BytesIO(mask_bytes)).convert("L")

        # Resize mask to match source if needed
        if mask_image.size != source_image.size:
            mask_image = mask_image.resize(source_image.size, Image.NEAREST)

        # Load inpainting pipeline from the same base model
        if self._inpaint_pipeline is None:
            self._inpaint_pipeline = AutoPipelineForInpainting.from_pipe(self.pipeline)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        # Match output resolution to source image
        source_width, source_height = source_image.size
        # SDXL needs dimensions divisible by 8
        target_width = (source_width // 8) * 8
        target_height = (source_height // 8) * 8

        output = self._inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image.resize((target_width, target_height)),
            mask_image=mask_image.resize((target_width, target_height), Image.NEAREST),
            width=target_width,
            height=target_height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        return output.images[0]

    def is_loaded(self) -> bool:
        return self._loaded
