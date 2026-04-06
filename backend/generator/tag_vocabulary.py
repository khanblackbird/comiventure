"""Tag vocabulary — the bridge between human descriptions and SDXL models.

SDXL anime models (Pony, Animagine, AAM) were trained on Danbooru/e621
tags. Freeform text like "wearing a blue dress" gets misinterpreted.
The correct tag is "blue_dress".

This module:
1. Defines the canonical tag vocabulary per field
2. Converts freeform text → closest valid tags
3. Provides model-specific quality/meta tag prefixes
4. Validates tags against the vocabulary

The vocabulary comes from what these models were actually trained on:
Danbooru (anime), e621 (furry), Derpibooru (pony).
"""
from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path

log = logging.getLogger(__name__)


# ── Canonical tag sets ─────────────────────────────────────────────────
# Each set maps a field name to tags the models actually understand.
# Tags use underscores for multi-word (Danbooru format).

POSES = {
    "standing", "sitting", "kneeling", "lying", "crouching", "squatting",
    "running", "walking", "jumping", "falling", "floating", "flying",
    "crawling", "fighting_stance", "dynamic_pose", "all_fours",
    "leaning_forward", "leaning_back", "on_back", "on_side", "on_stomach",
    "crossed_legs", "legs_apart", "seiza", "indian_style",
    "sitting_on_chair", "sitting_on_ground", "reclining",
    "arms_up", "arms_at_sides", "arms_behind_back", "crossed_arms",
    "hand_on_hip", "hands_on_hips", "outstretched_arms",
    "contrapposto", "stretching", "handstand", "midair",
}

EXPRESSIONS = {
    "smile", "grin", "light_smile", "smirk", "evil_smile",
    "angry", "annoyed", "scowl", "glaring", "clenched_teeth",
    "sad", "frown", "crying", "tears", "sobbing",
    "surprised", "shocked", "scared", "panicking", "worried",
    "happy", "excited", "laughing",
    "embarrassed", "blush", "flustered",
    "serious", "determined", "expressionless",
    "confused", "thinking", "pensive",
    "sleepy", "exhausted", "bored",
    "nervous", "frustrated", "disgusted",
    "smug", "pout", "open_mouth", "closed_eyes", "half-closed_eyes",
    "wink", "looking_at_viewer", "looking_away", "looking_up",
    "looking_down",
}

FRAMING = {
    "portrait", "upper_body", "cowboy_shot", "full_body", "close-up",
    "wide_shot", "lower_body", "profile",
    "from_above", "from_below", "from_side", "from_behind",
    "straight-on", "dutch_angle", "pov",
}

HAIR_COLORS = {
    "blonde_hair", "black_hair", "silver_hair", "white_hair",
    "pink_hair", "purple_hair", "red_hair", "blue_hair",
    "green_hair", "brown_hair", "grey_hair", "orange_hair",
    "multicolored_hair", "gradient_hair", "two-tone_hair",
}

HAIR_STYLES = {
    "long_hair", "short_hair", "medium_hair", "very_long_hair",
    "ponytail", "twintails", "braid", "twin_braids", "side_ponytail",
    "messy_hair", "flowing_hair", "bob_cut", "hime_cut", "pixie_cut",
    "hair_over_one_eye", "bangs", "blunt_bangs", "side_swept_bangs",
    "ahoge", "drill_hair", "hair_bun", "low_ponytail",
}

EYE_COLORS = {
    "blue_eyes", "red_eyes", "green_eyes", "brown_eyes",
    "yellow_eyes", "purple_eyes", "amber_eyes", "pink_eyes",
    "heterochromia", "glowing_eyes", "slit_pupils",
}

SPECIES = {
    "human", "elf", "pointy_ears", "demon_girl", "angel",
    "cat_ears", "cat_tail", "fox_ears", "fox_tail",
    "wolf_ears", "wolf_tail", "dog_ears", "dog_tail",
    "rabbit_ears", "rabbit_tail", "horse_ears",
    "horns", "wings", "angel_wings", "dragon_wings", "tail",
    "anthro", "furry", "kemonomimi",
}

CLOTHING = {
    "school_uniform", "serafuku", "military_uniform", "maid",
    "dress", "white_dress", "black_dress", "sundress", "wedding_dress",
    "kimono", "yukata", "gothic_lolita", "china_dress",
    "shirt", "t-shirt", "blouse", "sweater", "hoodie", "tank_top",
    "crop_top", "cardigan", "turtleneck", "off-shoulder",
    "skirt", "miniskirt", "pleated_skirt", "long_skirt",
    "shorts", "short_shorts", "pants", "jeans", "leggings",
    "jacket", "blazer", "coat", "cape", "cloak", "vest", "armor", "robe",
    "swimsuit", "bikini", "one-piece_swimsuit",
    "nude", "topless", "barefoot",
}

ACCESSORIES = {
    "hat", "ribbon", "bow", "hairband", "glasses", "sunglasses",
    "scarf", "choker", "necklace", "earrings", "bracelet",
    "gloves", "thighhighs", "kneehighs", "boots", "high_heels",
    "sneakers", "belt", "bag", "backpack", "headphones",
    "crown", "tiara", "mask", "collar",
}

ACTIONS = {
    "fighting", "punching", "kicking", "sword_fighting", "casting_spell",
    "reading", "eating", "drinking", "cooking", "sleeping",
    "singing", "dancing", "playing_instrument",
    "hugging", "holding_hands", "kissing",
    "pointing", "waving", "peace_sign", "thumbs_up",
    "holding_sword", "holding_book", "holding_cup", "holding_weapon",
}

# All known tags for validation
ALL_TAGS = (
    POSES | EXPRESSIONS | FRAMING | HAIR_COLORS | HAIR_STYLES
    | EYE_COLORS | SPECIES | CLOTHING | ACCESSORIES | ACTIONS
)


# ── Model profiles ────────────────────────────────────────────────────
# Each model needs: prompt format, quality tags, negative tags.
# The format determines how ALL prompts are built — generation,
# inpainting, everything. It follows the model, not the operation.

DANBOORU = "danbooru"     # comma-separated underscore tags
NATURAL = "natural"       # plain English sentences
E621 = "e621"             # like Danbooru with some vocab differences
PONY = "pony"             # Danbooru + score tags + source tags


MODEL_PROFILES = {
    "Lykon/AAM_XL_AnimeMix": {
        "format": DANBOORU,
        "positive": [
            "masterpiece", "best_quality", "amazing_quality",
        ],
        "negative": [
            "(low_quality, worst_quality:1.4)", "cgi", "text",
            "signature", "watermark", "extra_limbs",
        ],
    },
    "cagliostrolab/animagine-xl-3.1": {
        "format": DANBOORU,
        "positive": [
            "masterpiece", "best_quality", "very_aesthetic",
            "absurdres", "newest",
        ],
        "negative": [
            "worst_quality", "low_quality", "displeasing", "oldest",
        ],
    },
    "CitronLegacy/ponyDiffusionV6XL_Diffusers": {
        "format": PONY,
        "positive": [
            "score_9", "score_8_up", "score_7_up",
            "score_6_up", "source_anime",
        ],
        "negative": [
            "score_5", "score_4", "low_quality",
        ],
    },
    "John6666/nova-furry-xl-il-v120-sdxl": {
        "format": E621,
        "positive": [
            "masterpiece", "best_quality",
        ],
        "negative": [
            "low_quality", "worst_quality", "text", "watermark",
        ],
    },
    "John6666/autismmix-sdxl-autismmix-pony-sdxl": {
        "format": DANBOORU,
        "positive": [
            "masterpiece", "best_quality",
        ],
        "negative": [
            "(low_quality, worst_quality:1.4)", "text",
            "signature", "watermark",
        ],
    },
}

DEFAULT_PROFILE = {
    "format": DANBOORU,
    "positive": ["masterpiece", "best_quality"],
    "negative": [
        "(low_quality, worst_quality:1.4)", "text",
        "signature", "watermark",
    ],
}


def get_model_profile(model_id: str) -> dict:
    """Get the full profile for a model (format + quality tags)."""
    return MODEL_PROFILES.get(model_id, DEFAULT_PROFILE)


def get_quality_tags(model_id: str) -> dict:
    """Get quality prefix/suffix tags for a specific model."""
    profile = get_model_profile(model_id)
    return {"positive": profile["positive"], "negative": profile["negative"]}


def get_prompt_format(model_id: str) -> str:
    """Get the prompt format for a model: 'danbooru', 'pony', 'e621', or 'natural'."""
    return get_model_profile(model_id).get("format", DANBOORU)


def is_tag_model(model_id: str) -> bool:
    """Does this model expect tag-based prompts?"""
    return get_prompt_format(model_id) != NATURAL


# ── Tag normalization ─────────────────────────────────────────────────

def normalize_tag(text: str) -> str:
    """Convert freeform text to Danbooru tag format.

    "long hair" → "long_hair"
    "Blue Hair" → "blue_hair"
    "wearing a school uniform" → "school_uniform"
    """
    text = text.strip().lower()
    # Filter out non-descriptive values
    if text in ("", "none", "n/a", "unknown", "default", "empty"):
        return ""
    # Strip filler words
    for filler in ("wearing ", "with ", "a ", "an ", "the "):
        if text.startswith(filler):
            text = text[len(filler):]
    text = text.strip()
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    # Remove non-tag characters
    text = re.sub(r'[^a-z0-9_\-()]', '', text)
    return text


def normalize_tags(text: str) -> list[str]:
    """Split a comma/space-separated string into normalized tags."""
    if not text:
        return []
    # Split on commas first, then normalize each
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [normalize_tag(p) for p in parts if normalize_tag(p)]


def find_closest_tag(text: str, tag_set: set[str]) -> str:
    """Find the closest matching tag in a set.

    Tries exact match, then substring match, then returns normalized input.
    """
    normalized = normalize_tag(text)
    if normalized in tag_set:
        return normalized

    # Substring match — "sitting on a chair" → "sitting_on_chair"
    for tag in tag_set:
        if normalized in tag or tag in normalized:
            return tag

    # Partial word match — "ponytail hair" → "ponytail"
    words = normalized.split("_")
    for tag in tag_set:
        tag_words = tag.split("_")
        if any(w in tag_words for w in words):
            return tag

    # No match — return normalized input as-is (model might still know it)
    return normalized


def tags_for_appearance(
    species: str = "",
    hair_colour: str = "",
    hair_style: str = "",
    eye_colour: str = "",
    body_type: str = "",
    skin_tone: str = "",
    outfit: str = "",
    accessories: str = "",
    **kwargs,
) -> list[str]:
    """Convert appearance fields to ordered Danbooru tags."""
    tags = []

    if species:
        tags.append(find_closest_tag(species, SPECIES))

    if hair_colour:
        tag = find_closest_tag(hair_colour, HAIR_COLORS)
        if not tag.endswith("_hair"):
            tag = normalize_tag(hair_colour) + "_hair"
        tags.append(tag)

    if hair_style:
        tags.append(find_closest_tag(hair_style, HAIR_STYLES))

    if eye_colour:
        tag = find_closest_tag(eye_colour, EYE_COLORS)
        if not tag.endswith("_eyes"):
            tag = normalize_tag(eye_colour) + "_eyes"
        tags.append(tag)

    if body_type:
        tags.append(normalize_tag(body_type))

    if skin_tone:
        tags.append(normalize_tag(skin_tone))

    if outfit:
        for part in outfit.split(","):
            normalized = normalize_tag(part.strip())
            if normalized:
                tags.append(find_closest_tag(normalized, CLOTHING))

    if accessories:
        for part in accessories.split(","):
            normalized = normalize_tag(part.strip())
            if normalized:
                tags.append(find_closest_tag(normalized, ACCESSORIES))

    return [t for t in tags if t]


def tags_for_script(
    pose: str = "",
    action: str = "",
    emotion: str = "",
    outfit: str = "",
    direction: str = "",
) -> list[str]:
    """Convert script fields to Danbooru tags."""
    tags = []

    if pose:
        tags.append(find_closest_tag(pose, POSES))

    if action:
        tags.append(find_closest_tag(action, ACTIONS))

    if emotion:
        tags.append(find_closest_tag(emotion, EXPRESSIONS))

    if outfit:
        for part in outfit.split(","):
            normalized = normalize_tag(part.strip())
            if normalized:
                tags.append(find_closest_tag(normalized, CLOTHING))

    if direction:
        tags.append(find_closest_tag(direction, FRAMING))

    return [t for t in tags if t]


# ── Model-aware prompt formatting ─────────────────────────────────────

def format_prompt(tags: list[str], model_id: str) -> str:
    """Format a list of tags according to the model's expected format.

    For tag-based models: comma-separated Danbooru tags with quality prefix.
    For natural language models: join into a readable sentence.
    For Pony: prepend score + source tags.
    """
    profile = get_model_profile(model_id)
    fmt = profile.get("format", DANBOORU)
    quality = profile.get("positive", [])

    if fmt == NATURAL:
        return ", ".join(tags)

    # Danbooru / e621 / Pony: quality tags first, then content
    return ", ".join(quality + tags)


def format_negative(model_id: str, extra: list[str] = None) -> str:
    """Build the full negative prompt for a model."""
    profile = get_model_profile(model_id)
    parts = list(profile.get("negative", []))
    if extra:
        parts.extend(extra)
    seen = set()
    unique = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return ", ".join(unique)


# ── Auto-detection from safetensors metadata ──────────────────────────

def read_safetensors_metadata(file_path: str | Path) -> dict[str, str]:
    """Read __metadata__ from a safetensors file header.

    Only reads 8 bytes + JSON header — never touches the weights.
    A 6.5GB checkpoint reads in < 1ms.
    """
    try:
        with open(file_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                log.warning("Safetensors header too large: %d bytes", header_len)
                return {}
            header = json.loads(f.read(header_len))
        return header.get("__metadata__", {})
    except Exception as e:
        log.warning("Failed to read safetensors metadata: %s", e)
        return {}


def detect_format_from_metadata(metadata: dict[str, str]) -> str | None:
    """Detect prompt format from safetensors metadata.

    Checks kohya_ss training tags (ss_tag_frequency), base model name,
    and modelspec fields. Returns format string or None.
    """
    # 1. Check ss_tag_frequency — the actual training tags
    tag_freq_raw = metadata.get("ss_tag_frequency")
    if tag_freq_raw:
        try:
            tag_freq = json.loads(tag_freq_raw)
        except (json.JSONDecodeError, TypeError):
            tag_freq = {}

        all_tags = set()
        for folder_tags in tag_freq.values():
            if isinstance(folder_tags, dict):
                all_tags.update(folder_tags.keys())

        if all_tags:
            detected = _classify_tags(all_tags)
            if detected:
                log.info("Detected format '%s' from %d training tags", detected, len(all_tags))
                return detected

    # 2. Check base model name
    base_model = metadata.get("ss_sd_model_name", "").lower()
    if "pony" in base_model or "pdxl" in base_model:
        return PONY
    if "animagine" in base_model or "aam" in base_model:
        return DANBOORU
    if "e621" in base_model or "furry" in base_model:
        return E621

    # 3. Check modelspec title
    title = metadata.get("modelspec.title", "").lower()
    if "pony" in title:
        return PONY
    if "anime" in title or "danbooru" in title:
        return DANBOORU

    return None


def _classify_tags(tags: set[str]) -> str | None:
    """Classify a set of training tags into a format."""
    pony_markers = {"score_9", "score_8_up", "score_7_up",
                    "source_anime", "source_furry"}
    if tags & pony_markers:
        return PONY

    e621_markers = {"anthro", "feral", "rating_safe",
                    "rating_explicit", "canine", "equine"}
    if len(tags & e621_markers) >= 2:
        return E621

    danbooru_markers = {"1girl", "1boy", "solo", "highres",
                        "masterpiece", "looking_at_viewer"}
    if tags & danbooru_markers:
        return DANBOORU

    # Natural language heuristic: long tags, many spaces
    if tags:
        avg_len = sum(len(t) for t in tags) / len(tags)
        space_ratio = sum(1 for t in tags if " " in t) / len(tags)
        if avg_len > 20 or space_ratio > 0.5:
            return NATURAL

    return None


def detect_format_from_file(file_path: str | Path) -> str | None:
    """Auto-detect prompt format from a safetensors file.

    Reads the header metadata (< 1ms), checks for training tags,
    base model name, and modelspec fields. Returns format string
    or None if detection fails.
    """
    metadata = read_safetensors_metadata(file_path)
    if not metadata:
        return None
    return detect_format_from_metadata(metadata)


def auto_profile_for_file(file_path: str | Path) -> dict:
    """Build a model profile for an uploaded safetensors file.

    Reads metadata, detects format, returns a profile dict compatible
    with MODEL_PROFILES. Falls back to DEFAULT_PROFILE if detection fails.
    """
    metadata = read_safetensors_metadata(file_path)
    detected = detect_format_from_metadata(metadata) if metadata else None

    if detected == PONY:
        return {
            "format": PONY,
            "positive": [
                "score_9", "score_8_up", "score_7_up",
                "score_6_up", "source_anime",
            ],
            "negative": ["score_5", "score_4", "low_quality"],
            "detected_from": "metadata",
        }
    if detected == E621:
        return {
            "format": E621,
            "positive": ["masterpiece", "best_quality"],
            "negative": ["low_quality", "worst_quality", "text", "watermark"],
            "detected_from": "metadata",
        }
    if detected == NATURAL:
        return {
            "format": NATURAL,
            "positive": [],
            "negative": ["low quality, blurry, distorted"],
            "detected_from": "metadata",
        }
    if detected == DANBOORU:
        return {
            "format": DANBOORU,
            "positive": ["masterpiece", "best_quality"],
            "negative": [
                "(low_quality, worst_quality:1.4)", "text",
                "signature", "watermark",
            ],
            "detected_from": "metadata",
        }

    return {**DEFAULT_PROFILE, "detected_from": "fallback"}


def extract_tag_capabilities(file_path: str | Path) -> dict:
    """Extract what tag categories a model understands from its training data.

    Reads ss_tag_frequency, classifies each tag into a category,
    returns per-category tag lists sorted by frequency.

    This tells us: which fields matter for this model, and what
    vocabulary it expects per field.
    """
    metadata = read_safetensors_metadata(file_path)
    tag_freq_raw = metadata.get("ss_tag_frequency")
    if not tag_freq_raw:
        return {}

    try:
        tag_freq = json.loads(tag_freq_raw)
    except (json.JSONDecodeError, TypeError):
        return {}

    # Flatten all tags with frequencies
    flat: dict[str, int] = {}
    for folder_tags in tag_freq.values():
        if isinstance(folder_tags, dict):
            for tag, count in folder_tags.items():
                flat[tag] = flat.get(tag, 0) + (count if isinstance(count, int) else 1)

    if not flat:
        return {}

    # Classify each tag into a category
    category_map = {
        "pose": POSES,
        "expression": EXPRESSIONS,
        "framing": FRAMING,
        "hair_color": HAIR_COLORS,
        "hair_style": HAIR_STYLES,
        "eye_color": EYE_COLORS,
        "species": SPECIES,
        "clothing": CLOTHING,
        "accessories": ACCESSORIES,
        "action": ACTIONS,
    }

    capabilities: dict[str, list[tuple[str, int]]] = {
        cat: [] for cat in category_map
    }
    capabilities["other"] = []

    for tag, count in flat.items():
        matched = False
        for cat_name, cat_set in category_map.items():
            if tag in cat_set:
                capabilities[cat_name].append((tag, count))
                matched = True
                break
        if not matched:
            capabilities["other"].append((tag, count))

    # Sort each category by frequency (highest first)
    for cat in capabilities:
        capabilities[cat].sort(key=lambda x: x[1], reverse=True)

    # Build summary
    result = {}
    for cat, tag_list in capabilities.items():
        if tag_list:
            result[cat] = {
                "count": len(tag_list),
                "top_tags": [t[0] for t in tag_list[:20]],
                "total_frequency": sum(t[1] for t in tag_list),
            }

    return result


# Cached capabilities per model path
_capabilities_cache: dict[str, dict] = {}


def get_tag_capabilities(model_id: str) -> dict:
    """Get tag capabilities for a model. Cached after first read.

    For built-in HuggingFace models, returns default category sets.
    For local files, reads safetensors metadata.
    """
    if model_id in _capabilities_cache:
        return _capabilities_cache[model_id]

    # Try as a file path (local checkpoints / LoRAs)
    path = Path(model_id)
    if path.exists() and path.suffix == ".safetensors":
        caps = extract_tag_capabilities(path)
        if caps:
            _capabilities_cache[model_id] = caps
            log.info(
                "Extracted capabilities for %s: %s",
                path.name,
                {k: v["count"] for k, v in caps.items()},
            )
            return caps

    # Fallback: return our canonical tag sets as default capabilities
    _capabilities_cache[model_id] = {}
    return {}


def get_field_suggestions(model_id: str, field: str, limit: int = 15) -> list[str]:
    """Get suggested tags for a specific field based on the model's training data.

    If model has capabilities metadata, returns its top tags for that field.
    Otherwise returns our canonical tag set.
    """
    caps = get_tag_capabilities(model_id)

    # Map our field names to capability categories
    field_to_category = {
        "pose": "pose",
        "action": "action",
        "emotion": "expression",
        "expression": "expression",
        "outfit": "clothing",
        "direction": "framing",
        "framing": "framing",
        "species": "species",
        "hair_colour": "hair_color",
        "hair_style": "hair_style",
        "eye_colour": "eye_color",
        "accessories": "accessories",
    }

    category = field_to_category.get(field)
    if not category:
        return []

    # If we have model-specific data, use it
    if category in caps:
        return caps[category]["top_tags"][:limit]

    # Fallback to canonical tag sets
    canonical = {
        "pose": POSES,
        "expression": EXPRESSIONS,
        "framing": FRAMING,
        "hair_color": HAIR_COLORS,
        "hair_style": HAIR_STYLES,
        "eye_color": EYE_COLORS,
        "species": SPECIES,
        "clothing": CLOTHING,
        "accessories": ACCESSORIES,
        "action": ACTIONS,
    }
    tag_set = canonical.get(category, set())
    return sorted(tag_set)[:limit]
