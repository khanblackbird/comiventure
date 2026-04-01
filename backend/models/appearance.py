"""Character appearance — structured visual properties, reference image bank,
and training data for the per-character adapter layer.

Each character is essentially their own trainable entity:
- Structured properties define their visual identity
- Reference images form a training bank with labels
- The bank can be imported/exported between stories
- Edits feed back as new training samples

A reference image + its labels = one training pair for the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppearanceProperties:
    """Structured visual properties for a character."""
    species: str = ""
    body_type: str = ""
    height: str = ""
    skin_tone: str = ""
    hair_style: str = ""
    hair_colour: str = ""
    eye_colour: str = ""
    facial_features: str = ""
    outfit: str = ""
    accessories: str = ""
    art_style_notes: str = ""

    def to_prompt(self) -> str:
        parts = []
        if self.species:
            parts.append(self.species)
        if self.body_type:
            parts.append(f"{self.body_type} build")
        if self.height:
            parts.append(self.height)
        if self.skin_tone:
            parts.append(self.skin_tone)
        if self.hair_colour or self.hair_style:
            hair = " ".join(filter(None, [self.hair_colour, self.hair_style]))
            parts.append(f"{hair} hair")
        if self.eye_colour:
            parts.append(f"{self.eye_colour} eyes")
        if self.facial_features:
            parts.append(self.facial_features)
        if self.outfit:
            parts.append(f"wearing {self.outfit}")
        if self.accessories:
            parts.append(f"with {self.accessories}")
        if self.art_style_notes:
            parts.append(self.art_style_notes)
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "body_type": self.body_type,
            "height": self.height,
            "skin_tone": self.skin_tone,
            "hair_style": self.hair_style,
            "hair_colour": self.hair_colour,
            "eye_colour": self.eye_colour,
            "facial_features": self.facial_features,
            "outfit": self.outfit,
            "accessories": self.accessories,
            "art_style_notes": self.art_style_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AppearanceProperties:
        return cls(**{key: data.get(key, "") for key in cls.__dataclass_fields__})


@dataclass
class ReferenceImage:
    """A labeled reference image — one training sample for the adapter.

    content_hash: image data in ContentStore
    source: how this image was created
    accepted: binary training signal (True=positive, False=negative, None=unrated)
    caption: freeform text description of what's in the image
    pose: body pose (e.g. 'standing', 'sitting', 'action', 'portrait')
    expression: facial expression (e.g. 'happy', 'angry', 'neutral')
    angle: camera angle (e.g. 'front', 'side', '3/4', 'back')
    scene: scene context (e.g. 'outdoors forest', 'indoor cafe', 'battle')
    outfit_variant: if character has multiple outfits (e.g. 'casual', 'armor', 'formal')
    tags: additional freeform tags
    notes: user notes about this reference
    """
    content_hash: str
    source: str = "upload"          # 'upload', 'generated', 'edited', 'imported'
    accepted: Optional[bool] = None
    caption: str = ""
    pose: str = ""
    expression: str = ""
    angle: str = ""
    scene: str = ""
    outfit_variant: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_training_prompt(self) -> str:
        """Compose labels into a training caption for the adapter."""
        parts = []
        if self.caption:
            parts.append(self.caption)
        if self.pose:
            parts.append(self.pose)
        if self.expression:
            parts.append(f"{self.expression} expression")
        if self.angle:
            parts.append(f"{self.angle} view")
        if self.scene:
            parts.append(self.scene)
        if self.outfit_variant:
            parts.append(f"wearing {self.outfit_variant}")
        if self.tags:
            parts.extend(self.tags)
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "source": self.source,
            "accepted": self.accepted,
            "caption": self.caption,
            "pose": self.pose,
            "expression": self.expression,
            "angle": self.angle,
            "scene": self.scene,
            "outfit_variant": self.outfit_variant,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReferenceImage:
        return cls(
            content_hash=data["content_hash"],
            source=data.get("source", "upload"),
            accepted=data.get("accepted"),
            caption=data.get("caption", ""),
            pose=data.get("pose", ""),
            expression=data.get("expression", ""),
            angle=data.get("angle", ""),
            scene=data.get("scene", ""),
            outfit_variant=data.get("outfit_variant", ""),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


class Appearance:
    """Complete character appearance — properties + reference image bank.

    The bank is the character's training data:
    - Accepted references = positive training pairs
    - Rejected references = negative training pairs
    - Each reference's labels become the training caption

    Characters are portable — export the appearance (properties + bank)
    and import into another story. The adapter weights travel with them.
    """

    def __init__(self) -> None:
        self.properties = AppearanceProperties()
        self.references: list[ReferenceImage] = []
        self.adapter_hash: Optional[str] = None  # trained adapter weights in ContentStore

    def add_reference(
        self,
        content_hash: str,
        source: str = "upload",
        caption: str = "",
        pose: str = "",
        expression: str = "",
        angle: str = "",
        scene: str = "",
        outfit_variant: str = "",
        tags: list[str] | None = None,
    ) -> ReferenceImage:
        reference = ReferenceImage(
            content_hash=content_hash,
            source=source,
            caption=caption,
            pose=pose,
            expression=expression,
            angle=angle,
            scene=scene,
            outfit_variant=outfit_variant,
            tags=tags or [],
        )
        self.references.append(reference)
        return reference

    def get_reference(self, content_hash: str) -> ReferenceImage | None:
        for ref in self.references:
            if ref.content_hash == content_hash:
                return ref
        return None

    def remove_reference(self, content_hash: str) -> None:
        self.references = [ref for ref in self.references if ref.content_hash != content_hash]

    def accept_reference(self, content_hash: str) -> None:
        ref = self.get_reference(content_hash)
        if ref:
            ref.accepted = True

    def reject_reference(self, content_hash: str) -> None:
        ref = self.get_reference(content_hash)
        if ref:
            ref.accepted = False

    def accepted_references(self) -> list[ReferenceImage]:
        return [ref for ref in self.references if ref.accepted is True]

    def rejected_references(self) -> list[ReferenceImage]:
        return [ref for ref in self.references if ref.accepted is False]

    def unrated_references(self) -> list[ReferenceImage]:
        return [ref for ref in self.references if ref.accepted is None]

    def training_pairs(self) -> list[tuple[str, str]]:
        """Return (content_hash, caption) pairs for adapter training.
        Only accepted references are used as positive training data.
        """
        return [
            (ref.content_hash, ref.to_training_prompt())
            for ref in self.accepted_references()
            if ref.to_training_prompt()
        ]

    def to_prompt(self) -> str:
        return self.properties.to_prompt()

    def to_dict(self) -> dict:
        return {
            "properties": self.properties.to_dict(),
            "references": [ref.to_dict() for ref in self.references],
            "adapter_hash": self.adapter_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Appearance:
        appearance = cls()
        appearance.properties = AppearanceProperties.from_dict(data.get("properties", {}))
        appearance.references = [
            ReferenceImage.from_dict(ref) for ref in data.get("references", [])
        ]
        appearance.adapter_hash = data.get("adapter_hash")
        return appearance
