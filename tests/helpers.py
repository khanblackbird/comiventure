"""Shared test helpers — the ONLY correct way to build test hierarchies.

Every test that needs model objects must use these helpers.
Direct construction (Script(), Panel(), etc.) is impossible.
"""
from backend.models import Story, Character


def make_story(
    story_id: str = "s1",
    title: str = "Test Story",
    character_name: str = "Luna",
    character_id: str = "c1",
    appearance_prompt: str = "blue-haired girl",
    chapter_title: str = "Chapter One",
    chapter_synopsis: str = "the beginning",
) -> Story:
    """Create a minimum valid story with one of everything.
    Returns a story with: 1 character, 1 chapter, 1 page, 1 panel, 1 script.
    """
    story = Story(story_id, title)
    char = Character(character_id, character_name, appearance_prompt=appearance_prompt)
    story.add_character(char)
    story.create_chapter(chapter_title, [character_id], synopsis=chapter_synopsis)
    return story


def make_two_character_story() -> Story:
    """Story with two characters sharing a chapter."""
    story = Story("s1", "Test Story")
    story.add_character(Character("c1", "Luna", appearance_prompt="blue-haired girl"))
    story.add_character(Character("c2", "Rex", appearance_prompt="tall wolf anthro"))
    story.create_chapter("The Meeting", ["c1", "c2"], synopsis="they meet")
    return story


def get_chain(story: Story):
    """Extract the first complete chain from a story (skips solo chapters)."""
    chapters = [ch for ch in story.chapters.values() if not ch.is_solo]
    if not chapters:
        # Fall back to any chapter
        chapters = list(story.chapters.values())
    chapter = chapters[0]
    page = chapter.pages[0]
    panel = page.panels[0]
    script = list(panel.scripts.values())[0]
    character = story.get_character(script.character_id)
    return {
        "story": story,
        "character": character,
        "chapter": chapter,
        "page": page,
        "panel": panel,
        "script": script,
    }
