"""Save and load stories as .cvn files (ZIP archives).

A .cvn file contains:
- story.json: the full object hierarchy serialized
- content/: all referenced assets (images, video) by content hash

This keeps everything in one portable file. The content store's
hash-based addressing means assets map directly to filenames.
"""
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

from .content_store import ContentStore
from .story import Story
from .character import Character
from .chapter import Chapter
from .page import Page
from .panel import Panel
from .script import Script


def save_story(story: Story, content_store: ContentStore, file_path: str) -> str:
    """Save a story and all its assets to a .cvn file.
    Returns the path to the saved file.
    """
    file_path = _ensure_extension(file_path)

    # Auto-repair then validate integrity before saving
    story.repair()
    story.validate_or_raise()

    # Collect all content hashes referenced by panels
    content_hashes = _collect_content_hashes(story)

    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as archive:
        # Write the hierarchy as JSON
        story_data = story.to_dict()
        archive.writestr("story.json", json.dumps(story_data, indent=2))

        # Write each referenced asset into content/
        for content_hash in content_hashes:
            meta = content_store.get_meta(content_hash)
            if meta:
                asset_data = content_store.retrieve(content_hash)
                if asset_data:
                    extension = Path(meta.file_path).suffix
                    archive.writestr(
                        f"content/{content_hash}{extension}",
                        asset_data,
                    )

    return file_path


def load_story(file_path: str, content_store: ContentStore) -> Story:
    """Load a story and its assets from a .cvn file.
    Assets are imported into the content store.
    Returns the reconstructed Story with full hierarchy wired up.
    """
    with zipfile.ZipFile(file_path, "r") as archive:
        story_json = json.loads(archive.read("story.json"))

        # Import assets into content store
        content_files = [
            name for name in archive.namelist()
            if name.startswith("content/") and not name.endswith("/")
        ]
        for content_file in content_files:
            asset_data = archive.read(content_file)
            filename = Path(content_file).stem  # the hash
            extension = Path(content_file).suffix
            content_type = _type_for_extension(extension)
            content_store.store(asset_data, content_type)

    story = _rebuild_story(story_json)

    # Validate integrity after loading
    story.validate_or_raise()

    return story


def _ensure_extension(file_path: str) -> str:
    if not file_path.endswith(".cvn"):
        file_path += ".cvn"
    return file_path


def _collect_content_hashes(story: Story) -> set[str]:
    """Walk the hierarchy and collect all content hashes — panels and character references."""
    hashes = set()

    # Panel images and videos
    for chapter in story.chapters.values():
        for page in chapter.pages:
            for panel in page.panels:
                if panel.image_hash:
                    hashes.add(panel.image_hash)
                if panel.video_hash:
                    hashes.add(panel.video_hash)

    # Character reference images
    for character in story.characters.values():
        for reference in character.appearance.references:
            hashes.add(reference.content_hash)

    return hashes


def _rebuild_story(data: dict) -> Story:
    """Reconstruct the full object hierarchy from serialized JSON.
    Wires up all parent-child relationships and event emission.
    """
    story = Story(
        story_id=data["story_id"],
        title=data["title"],
        synopsis=data.get("synopsis", ""),
        art_style=data.get("art_style", ""),
        genre=data.get("genre", ""),
        negative_prompt=data.get("negative_prompt", ""),
    )
    story.style_loras = data.get("style_loras", [])
    story.style_references = data.get("style_references", [])

    # First pass: create all chapters
    chapters_by_id: dict[str, Chapter] = {}
    for chapter_id, chapter_data in data.get("chapters", {}).items():
        chapter = Chapter(
            chapter_id=chapter_data["chapter_id"],
            title=chapter_data.get("title", ""),
            synopsis=chapter_data.get("synopsis", ""),
            default_location=chapter_data.get("default_location", ""),
            default_time_of_day=chapter_data.get("default_time_of_day", ""),
            is_solo=chapter_data.get("is_solo", False),
        )
        chapter.negative_prompt = chapter_data.get("negative_prompt", "")
        # Restore character_ids directly from serialized data
        chapter.character_ids = chapter_data.get("character_ids", [])

        # Rebuild pages
        for page_data in chapter_data.get("pages", []):
            page = Page(
                page_id=page_data["page_id"],
                page_number=page_data.get("page_number", 0),
                layout_template=page_data.get("layout_template", "auto"),
                setting=page_data.get("setting", ""),
                mood=page_data.get("mood", ""),
                action_context=page_data.get("action_context", ""),
                time_of_day=page_data.get("time_of_day", ""),
                weather=page_data.get("weather", ""),
                lighting=page_data.get("lighting", ""),
            )
            page.negative_prompt = page_data.get("negative_prompt", "")

            # Rebuild panels
            for panel_data in page_data.get("panels", []):
                panel = Panel(
                    panel_id=panel_data["panel_id"],
                    image_hash=panel_data.get("image_hash"),
                    video_hash=panel_data.get("video_hash"),
                    narration=panel_data.get("narration", ""),
                    shot_type=panel_data.get("shot_type", ""),
                )
                panel.source = panel_data.get("source", "empty")
                panel.negative_prompt = panel_data.get("negative_prompt", "")

                # Rebuild scripts
                for character_id, script_data in panel_data.get("scripts", {}).items():
                    script = Script(
                        script_id=script_data["script_id"],
                        character_id=script_data["character_id"],
                        dialogue=script_data.get("dialogue", ""),
                        action=script_data.get("action", ""),
                        direction=script_data.get("direction", ""),
                        emotion=script_data.get("emotion", ""),
                        pose=script_data.get("pose", ""),
                        outfit=script_data.get("outfit", ""),
                    )
                    script.source = script_data.get("source", "empty")
                    script.negative_prompt = script_data.get("negative_prompt", "")
                    panel.add_script(script)
                    story.register_script(script)

                page.add_panel(panel)
                story.register_panel(panel)

            chapter.add_page(page)
            story.register_page(page)

        story.add_chapter(chapter)
        chapters_by_id[chapter_id] = chapter

    # Second pass: create characters and bind to chapters
    for character_id, character_data in data.get("characters", {}).items():
        character = Character(
            character_id=character_data["character_id"],
            name=character_data["name"],
            description=character_data.get("description", ""),
            personality_prompt=character_data.get("personality_prompt", ""),
            portrait_path=character_data.get("portrait_path"),
            is_temporary=character_data.get("is_temporary", False),
            page_scope=character_data.get("page_scope"),
        )

        # Restore structured appearance if present
        appearance_data = character_data.get("appearance")
        if appearance_data:
            from .appearance import Appearance
            character.appearance = Appearance.from_dict(appearance_data)
        elif character_data.get("appearance_prompt"):
            character.appearance.properties.art_style_notes = character_data["appearance_prompt"]

        # Restore profile if present
        profile_data = character_data.get("profile")
        if profile_data:
            from .profile import Profile
            character.profile = Profile.from_dict(profile_data)

        # Restore conversations
        character.conversations = character_data.get("conversations", [])
        character.negative_prompt = character_data.get("negative_prompt", "")

        # Wire character to story WITHOUT cascade — data already loaded
        character.set_parent(story)
        character.on("character_updated", story._on_character_updated)
        story.characters[character.character_id] = character
        story.register(character.character_id, character)

        # Bind character to their chapters WITHOUT cascade
        for chapter_id in character_data.get("chapters", []):
            chapter = chapters_by_id.get(chapter_id)
            if chapter:
                chapter.set_parent(character)
                if character.character_id not in chapter.character_ids:
                    chapter.character_ids.append(character.character_id)
                chapter.on("chapter_updated", character._on_chapter_updated)
                character.chapters.append(chapter)

    # Final registration pass — catch anything missed
    for chapter in story.chapters.values():
        story._register_cascade(chapter)

    return story


def _type_for_extension(extension: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".txt": "text/plain",
    }.get(extension, "application/octet-stream")
