"""Character profile — biographical and physical details.

These feed into the LLM for:
- In-character dialogue generation
- Prompt composition (the LLM knows the character has a gold tattoo on their rump)
- Continuity checking (adversarial — catch when details are wrong)

Separate from Appearance which is structured for image generation.
Profile is narrative — how the character IS, not just what they look like.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysicalTraits:
    """What the character looks like — narrative description, not prompt tokens."""
    body: str = ""              # "lean muscular build, 5'8"
    face: str = ""              # "sharp jawline, amber eyes, scar across left cheek"
    distinguishing_marks: str = ""  # "gold tattoo on left rump, notched ear"
    hair_fur: str = ""          # "black fur with orange tabby stripes"
    voice: str = ""             # "low raspy voice with a slight drawl"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> PhysicalTraits:
        return cls(**{k: data.get(k, "") for k in cls.__dataclass_fields__})


@dataclass
class Outfit:
    """A named outfit/look the character wears."""
    name: str = ""              # "casual", "armor", "formal"
    description: str = ""       # "torn leather jacket, no shirt, ripped jeans"
    is_default: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "is_default": self.is_default}

    @classmethod
    def from_dict(cls, data: dict) -> Outfit:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            is_default=data.get("is_default", False),
        )


class Profile:
    """Full character profile — who they are, not just what they look like.

    biography: backstory, history, motivations
    personality: how they act, speak, react
    physical: body, face, marks, voice
    outfits: named looks they switch between
    tendencies: habitual actions/expressions ("always smirks", "fidgets when nervous")
    expressions: common emotional states and how they manifest
    relationships: how they relate to other characters
    """

    def __init__(self) -> None:
        self.biography: str = ""
        self.personality: str = ""
        self.physical = PhysicalTraits()
        self.outfits: list[Outfit] = []
        self.tendencies: list[str] = []
        self.expressions: dict[str, str] = {}  # emotion -> how it looks: "angry" -> "ears flatten, tail lashes"
        self.relationships: dict[str, str] = {}  # character_id -> description: "c2" -> "rival turned lover"
        self.notes: str = ""  # freeform user notes

    def default_outfit(self) -> Outfit | None:
        for outfit in self.outfits:
            if outfit.is_default:
                return outfit
        return self.outfits[0] if self.outfits else None

    def add_outfit(self, name: str, description: str, is_default: bool = False) -> Outfit:
        if is_default:
            for outfit in self.outfits:
                outfit.is_default = False
        outfit = Outfit(name=name, description=description, is_default=is_default)
        self.outfits.append(outfit)
        return outfit

    def set_expression(self, emotion: str, description: str) -> None:
        """e.g. set_expression("angry", "ears flatten back, tail lashes, bares teeth")"""
        self.expressions[emotion] = description

    def get_expression(self, emotion: str) -> str:
        """Get how this emotion manifests for this character."""
        return self.expressions.get(emotion, "")

    def set_relationship(self, character_id: str, description: str) -> None:
        self.relationships[character_id] = description

    def to_llm_context(self) -> str:
        """Compose the full profile into a text block for the LLM system prompt."""
        parts = []
        if self.biography:
            parts.append(f"Biography: {self.biography}")
        if self.personality:
            parts.append(f"Personality: {self.personality}")
        if self.physical.body:
            parts.append(f"Body: {self.physical.body}")
        if self.physical.face:
            parts.append(f"Face: {self.physical.face}")
        if self.physical.distinguishing_marks:
            parts.append(f"Distinguishing marks: {self.physical.distinguishing_marks}")
        if self.physical.hair_fur:
            parts.append(f"Hair/Fur: {self.physical.hair_fur}")
        if self.physical.voice:
            parts.append(f"Voice: {self.physical.voice}")
        outfit = self.default_outfit()
        if outfit:
            parts.append(f"Current outfit: {outfit.description}")
        if self.tendencies:
            parts.append(f"Tendencies: {', '.join(self.tendencies)}")
        if self.expressions:
            expr_parts = [f"{k}: {v}" for k, v in self.expressions.items()]
            parts.append(f"Expressions: {'; '.join(expr_parts)}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "biography": self.biography,
            "personality": self.personality,
            "physical": self.physical.to_dict(),
            "outfits": [o.to_dict() for o in self.outfits],
            "tendencies": self.tendencies,
            "expressions": self.expressions,
            "relationships": self.relationships,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        profile = cls()
        profile.biography = data.get("biography", "")
        profile.personality = data.get("personality", "")
        profile.physical = PhysicalTraits.from_dict(data.get("physical", {}))
        profile.outfits = [Outfit.from_dict(o) for o in data.get("outfits", [])]
        profile.tendencies = data.get("tendencies", [])
        profile.expressions = data.get("expressions", {})
        profile.relationships = data.get("relationships", {})
        profile.notes = data.get("notes", "")
        return profile
