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

import logging
import re

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


# ── Model-specific quality prefixes ───────────────────────────────────

MODEL_QUALITY_TAGS = {
    "Lykon/AAM_XL_AnimeMix": {
        "positive": [
            "masterpiece", "best_quality", "amazing_quality",
        ],
        "negative": [
            "(low_quality, worst_quality:1.4)", "cgi", "text",
            "signature", "watermark", "extra_limbs",
        ],
    },
    "cagliostrolab/animagine-xl-3.1": {
        "positive": [
            "masterpiece", "best_quality", "very_aesthetic",
            "absurdres", "newest",
        ],
        "negative": [
            "worst_quality", "low_quality", "displeasing", "oldest",
        ],
    },
    "CitronLegacy/ponyDiffusionV6XL_Diffusers": {
        "positive": [
            "score_9", "score_8_up", "score_7_up",
            "score_6_up", "source_anime",
        ],
        "negative": [
            "score_5", "score_4", "low_quality",
        ],
    },
}

# Fallback for custom checkpoints and unknown models
DEFAULT_QUALITY_TAGS = {
    "positive": ["masterpiece", "best_quality"],
    "negative": [
        "(low_quality, worst_quality:1.4)", "text",
        "signature", "watermark",
    ],
}


def get_quality_tags(model_id: str) -> dict:
    """Get quality prefix/suffix tags for a specific model."""
    return MODEL_QUALITY_TAGS.get(model_id, DEFAULT_QUALITY_TAGS)


# ── Tag normalization ─────────────────────────────────────────────────

def normalize_tag(text: str) -> str:
    """Convert freeform text to Danbooru tag format.

    "long hair" → "long_hair"
    "Blue Hair" → "blue_hair"
    "wearing a school uniform" → "school_uniform"
    """
    text = text.strip().lower()
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
            tags.append(find_closest_tag(part.strip(), CLOTHING))

    if accessories:
        for part in accessories.split(","):
            tags.append(find_closest_tag(part.strip(), ACCESSORIES))

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
            tags.append(find_closest_tag(part.strip(), CLOTHING))

    if direction:
        tags.append(find_closest_tag(direction, FRAMING))

    return [t for t in tags if t]
