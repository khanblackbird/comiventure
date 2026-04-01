"""End-to-end tests for the Comiventure API using FastAPI TestClient.

These tests exercise the full HTTP API: creating stories, characters,
chapters, pages, panels, and scripts, and verifying their interactions.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.api import routes
from backend.models.story import Story
from backend.models.content_store import ContentStore


@pytest.fixture(autouse=True)
def fresh_story(tmp_path):
    """Reset the global story and content store before each test."""
    routes.content_store = ContentStore(str(tmp_path / "content"))
    routes.story = Story("test-story", "Test Story")
    yield
    routes.story = None


@pytest.fixture
def client():
    return TestClient(app)


# ---------- helpers ----------


def create_character(client, name="Luna", appearance_prompt="blue-haired girl"):
    """Create a character and return the response JSON."""
    response = client.post("/api/characters", json={
        "name": name,
        "appearance_prompt": appearance_prompt,
    })
    assert response.status_code == 200
    return response.json()


def create_chapter(client, title, character_ids, synopsis="", default_location="", default_time_of_day=""):
    """Create a chapter and return the response JSON."""
    response = client.post("/api/chapters", json={
        "title": title,
        "character_ids": character_ids,
        "synopsis": synopsis,
        "default_location": default_location,
        "default_time_of_day": default_time_of_day,
    })
    assert response.status_code == 200
    return response.json()


def create_page(client, chapter_id, layout_template="auto"):
    """Create a page and return the response JSON."""
    response = client.post("/api/pages", json={
        "chapter_id": chapter_id,
        "layout_template": layout_template,
    })
    assert response.status_code == 200
    return response.json()


def create_panel(client, page_id):
    """Create a panel and return the response JSON."""
    response = client.post("/api/panels", json={"page_id": page_id})
    assert response.status_code == 200
    return response.json()


# ---------- Story creation and hierarchy ----------


class TestStoryCreationAndHierarchy:
    def test_create_story_add_characters_create_chapter(self, client):
        """Full hierarchy: story -> characters -> chapter -> page -> panel -> scripts."""
        luna = create_character(client, "Luna", "blue-haired girl")
        rex = create_character(client, "Rex", "tall wolf anthro")

        chapter = create_chapter(
            client,
            "The Meeting",
            [luna["character_id"], rex["character_id"]],
            synopsis="They meet for the first time",
        )

        assert chapter["title"] == "The Meeting"
        assert luna["character_id"] in chapter["character_ids"]
        assert rex["character_id"] in chapter["character_ids"]

        # Chapter auto-creates page -> panel -> scripts
        assert len(chapter["pages"]) >= 1
        first_page = chapter["pages"][0]
        assert len(first_page["panels"]) >= 1
        first_panel = first_page["panels"][0]
        assert len(first_panel["scripts"]) >= 2

    def test_get_story_returns_full_data(self, client):
        luna = create_character(client, "Luna")
        create_chapter(client, "Chapter 1", [luna["character_id"]])

        response = client.get("/api/story")
        assert response.status_code == 200
        story_data = response.json()
        assert story_data["title"] == "Test Story"
        assert len(story_data["characters"]) >= 1
        assert len(story_data["chapters"]) >= 1

    def test_list_characters(self, client):
        create_character(client, "Luna")
        create_character(client, "Rex")

        response = client.get("/api/characters")
        assert response.status_code == 200
        characters = response.json()
        names = {character_data["name"] for character_data in characters.values()}
        assert "Luna" in names
        assert "Rex" in names

    def test_list_chapters(self, client):
        luna = create_character(client, "Luna")
        create_chapter(client, "Chapter 1", [luna["character_id"]])
        create_chapter(client, "Chapter 2", [luna["character_id"]])

        response = client.get("/api/chapters")
        assert response.status_code == 200
        chapters = response.json()
        titles = {chapter_data["title"] for chapter_data in chapters.values()}
        assert "Chapter 1" in titles
        assert "Chapter 2" in titles


# ---------- Pages, panels, scripts CRUD ----------


class TestPagePanelScriptCrud:
    def test_create_page_in_chapter(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        chapter_id = chapter["chapter_id"]

        page = create_page(client, chapter_id)
        assert page["page_id"] is not None
        assert len(page["panels"]) >= 1

    def test_create_panel_in_page(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        page_id = chapter["pages"][0]["page_id"]

        panel = create_panel(client, page_id)
        assert panel["panel_id"] is not None
        assert luna["character_id"] in panel["scripts"]

    def test_create_and_update_script(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        first_panel = chapter["pages"][0]["panels"][0]
        script_data = first_panel["scripts"][luna["character_id"]]
        script_id = script_data["script_id"]

        # Update the script
        response = client.put(f"/api/scripts/{script_id}", json={
            "dialogue": "Hello there!",
            "action": "waves hand",
            "emotion": "happy",
        })
        assert response.status_code == 200
        updated = response.json()
        assert updated["dialogue"] == "Hello there!"
        assert updated["action"] == "waves hand"
        assert updated["emotion"] == "happy"

    def test_update_panel_shot_type_and_narration(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        panel_id = chapter["pages"][0]["panels"][0]["panel_id"]

        response = client.put(f"/api/panels/{panel_id}", json={
            "shot_type": "close-up",
            "narration": "The wind howled...",
        })
        assert response.status_code == 200
        updated = response.json()
        assert updated["shot_type"] == "close-up"
        assert updated["narration"] == "The wind howled..."


# ---------- Page context inheritance ----------


class TestPageContextInheritance:
    def test_new_page_inherits_previous_page_setting_and_mood(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(
            client, "Chapter 1", [luna["character_id"]],
            default_location="enchanted forest",
        )
        chapter_id = chapter["chapter_id"]
        first_page_id = chapter["pages"][0]["page_id"]

        # Set setting and mood on first page
        client.put(f"/api/pages/{first_page_id}", json={
            "setting": "dark cave",
            "mood": "tense",
            "weather": "rain",
            "lighting": "firelight",
        })

        # Create a second page -- should inherit from the first
        second_page = create_page(client, chapter_id)
        assert second_page["setting"] == "dark cave"
        assert second_page["mood"] == "tense"
        assert second_page["weather"] == "rain"
        assert second_page["lighting"] == "firelight"

    def test_first_page_inherits_chapter_defaults(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(
            client, "Chapter 1", [luna["character_id"]],
            default_location="space station",
            default_time_of_day="night",
        )

        first_page = chapter["pages"][0]
        assert first_page["setting"] == "space station"
        assert first_page["time_of_day"] == "night"


# ---------- Solo chapter ----------


class TestSoloChapter:
    def test_get_solo_chapter_creates_one(self, client):
        luna = create_character(client, "Luna")
        character_id = luna["character_id"]

        response = client.get(f"/api/characters/{character_id}/solo-chapter")
        assert response.status_code == 200
        solo = response.json()
        assert solo["is_solo"] is True
        assert character_id in solo["character_ids"]
        assert len(solo["pages"]) >= 1

    def test_solo_chapter_is_idempotent(self, client):
        luna = create_character(client, "Luna")
        character_id = luna["character_id"]

        first_response = client.get(f"/api/characters/{character_id}/solo-chapter")
        second_response = client.get(f"/api/characters/{character_id}/solo-chapter")

        assert first_response.json()["chapter_id"] == second_response.json()["chapter_id"]


# ---------- Save / Load roundtrip ----------


class TestSaveLoadRoundtrip:
    def test_save_and_load_via_api(self, client):
        # Build a populated story
        luna = create_character(client, "Luna", "blue-haired girl")
        rex = create_character(client, "Rex", "tall wolf anthro")
        chapter = create_chapter(
            client,
            "The Meeting",
            [luna["character_id"], rex["character_id"]],
            synopsis="They meet",
        )

        # Update a script with dialogue
        first_panel = chapter["pages"][0]["panels"][0]
        script_id = first_panel["scripts"][luna["character_id"]]["script_id"]
        client.put(f"/api/scripts/{script_id}", json={"dialogue": "Hi Rex!"})

        # Save
        save_response = client.post("/api/story/save")
        assert save_response.status_code == 200
        save_data = save_response.json()
        assert "filename" in save_data

        # Load the saved file
        load_response = client.post(f"/api/story/load/{save_data['filename']}")
        assert load_response.status_code == 200
        loaded_story = load_response.json()

        # Verify structure survived the roundtrip
        assert loaded_story["title"] == "Test Story"
        assert len(loaded_story["characters"]) == 2
        character_names = {
            character_data["name"]
            for character_data in loaded_story["characters"].values()
        }
        assert "Luna" in character_names
        assert "Rex" in character_names

        # Verify chapters survived
        assert len(loaded_story["chapters"]) >= 1
        loaded_chapters = list(loaded_story["chapters"].values())
        meeting_chapters = [
            chapter_data for chapter_data in loaded_chapters
            if chapter_data["title"] == "The Meeting"
        ]
        assert len(meeting_chapters) == 1
        loaded_chapter = meeting_chapters[0]
        assert len(loaded_chapter["pages"]) >= 1
        assert len(loaded_chapter["pages"][0]["panels"]) >= 1


# ---------- Update story, chapter, page properties ----------


class TestPropertyUpdates:
    def test_update_story_art_style(self, client):
        response = client.put("/api/story", json={"art_style": "watercolor"})
        assert response.status_code == 200
        assert response.json()["art_style"] == "watercolor"

        # Verify it persists on subsequent GET
        story_data = client.get("/api/story").json()
        assert story_data["art_style"] == "watercolor"

    def test_update_story_genre(self, client):
        response = client.put("/api/story", json={"genre": "sci-fi"})
        assert response.status_code == 200
        assert response.json()["genre"] == "sci-fi"

    def test_update_chapter_location(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        chapter_id = chapter["chapter_id"]

        response = client.put(f"/api/chapters/{chapter_id}", json={
            "default_location": "ancient ruins",
        })
        assert response.status_code == 200
        assert response.json()["default_location"] == "ancient ruins"

    def test_update_chapter_time_of_day(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        chapter_id = chapter["chapter_id"]

        response = client.put(f"/api/chapters/{chapter_id}", json={
            "default_time_of_day": "dawn",
        })
        assert response.status_code == 200
        assert response.json()["default_time_of_day"] == "dawn"

    def test_update_page_weather(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        page_id = chapter["pages"][0]["page_id"]

        response = client.put(f"/api/pages/{page_id}", json={"weather": "snow"})
        assert response.status_code == 200
        assert response.json()["weather"] == "snow"

    def test_update_page_multiple_fields(self, client):
        luna = create_character(client, "Luna")
        chapter = create_chapter(client, "Chapter 1", [luna["character_id"]])
        page_id = chapter["pages"][0]["page_id"]

        response = client.put(f"/api/pages/{page_id}", json={
            "setting": "throne room",
            "mood": "dramatic",
            "time_of_day": "dusk",
            "weather": "storm",
            "lighting": "candlelight",
            "action_context": "confrontation",
        })
        assert response.status_code == 200
        page_data = response.json()
        assert page_data["setting"] == "throne room"
        assert page_data["mood"] == "dramatic"
        assert page_data["time_of_day"] == "dusk"
        assert page_data["weather"] == "storm"
        assert page_data["lighting"] == "candlelight"
        assert page_data["action_context"] == "confrontation"


# ---------- Character appearance update ----------


class TestCharacterAppearanceUpdate:
    def test_update_appearance_via_api(self, client):
        luna = create_character(client, "Luna")
        character_id = luna["character_id"]

        response = client.put(f"/api/characters/{character_id}/appearance", json={
            "hair_colour": "silver",
            "eye_colour": "violet",
            "body_type": "slender",
        })
        assert response.status_code == 200
        appearance = response.json()
        assert appearance["properties"]["hair_colour"] == "silver"
        assert appearance["properties"]["eye_colour"] == "violet"
        assert appearance["properties"]["body_type"] == "slender"

    def test_update_character_name_and_description(self, client):
        luna = create_character(client, "Luna")
        character_id = luna["character_id"]

        response = client.put(f"/api/characters/{character_id}", json={
            "name": "Luna Starweaver",
            "description": "A mysterious mage",
        })
        assert response.status_code == 200
        updated = response.json()
        assert updated["name"] == "Luna Starweaver"
        assert updated["description"] == "A mysterious mage"

    def test_appearance_persists_on_get(self, client):
        luna = create_character(client, "Luna")
        character_id = luna["character_id"]

        client.put(f"/api/characters/{character_id}/appearance", json={
            "hair_style": "long braids",
        })

        response = client.get(f"/api/characters/{character_id}/appearance")
        assert response.status_code == 200
        assert response.json()["properties"]["hair_style"] == "long braids"


# ---------- Full lifecycle ----------


class TestFullLifecycle:
    def test_create_populate_save_load_verify(self, client):
        """Complete lifecycle: create -> populate -> save -> load -> verify."""

        # 1. Create characters
        luna = create_character(client, "Luna", "blue-haired girl")
        rex = create_character(client, "Rex", "tall wolf anthro")
        luna_id = luna["character_id"]
        rex_id = rex["character_id"]

        # 2. Update story metadata
        client.put("/api/story", json={
            "art_style": "manga",
            "genre": "fantasy",
            "synopsis": "Two heroes embark on a quest",
        })

        # 3. Update character appearances
        client.put(f"/api/characters/{luna_id}/appearance", json={
            "hair_colour": "blue",
            "eye_colour": "green",
        })

        # 4. Create chapter with location
        chapter = create_chapter(
            client, "The Quest Begins",
            [luna_id, rex_id],
            synopsis="Heroes set out",
            default_location="village square",
            default_time_of_day="dawn",
        )
        chapter_id = chapter["chapter_id"]

        # 5. Update first page context
        first_page_id = chapter["pages"][0]["page_id"]
        client.put(f"/api/pages/{first_page_id}", json={
            "mood": "hopeful",
            "weather": "clear",
        })

        # 6. Update scripts with dialogue
        first_panel = chapter["pages"][0]["panels"][0]
        luna_script_id = first_panel["scripts"][luna_id]["script_id"]
        rex_script_id = first_panel["scripts"][rex_id]["script_id"]

        client.put(f"/api/scripts/{luna_script_id}", json={
            "dialogue": "Ready to go?",
            "emotion": "excited",
            "action": "adjusts backpack",
        })
        client.put(f"/api/scripts/{rex_script_id}", json={
            "dialogue": "Always.",
            "emotion": "calm",
            "pose": "standing tall",
        })

        # 7. Add a second page
        second_page = create_page(client, chapter_id)
        second_page_id = second_page["page_id"]

        # Second page should inherit from first page
        assert second_page["mood"] == "hopeful"
        assert second_page["weather"] == "clear"

        # 8. Add a panel to the second page
        new_panel = create_panel(client, second_page_id)
        assert luna_id in new_panel["scripts"]
        assert rex_id in new_panel["scripts"]

        # 9. Save
        save_response = client.post("/api/story/save")
        assert save_response.status_code == 200
        filename = save_response.json()["filename"]

        # 10. Load into a fresh story
        load_response = client.post(f"/api/story/load/{filename}")
        assert load_response.status_code == 200
        loaded = load_response.json()

        # 11. Verify everything survived
        assert loaded["art_style"] == "manga"
        assert loaded["genre"] == "fantasy"
        assert loaded["synopsis"] == "Two heroes embark on a quest"
        assert len(loaded["characters"]) == 2

        loaded_character_names = {
            character_data["name"]
            for character_data in loaded["characters"].values()
        }
        assert "Luna" in loaded_character_names
        assert "Rex" in loaded_character_names

        loaded_chapters = list(loaded["chapters"].values())
        quest_chapters = [
            chapter_data for chapter_data in loaded_chapters
            if chapter_data["title"] == "The Quest Begins"
        ]
        assert len(quest_chapters) == 1
        loaded_chapter = quest_chapters[0]
        assert loaded_chapter["synopsis"] == "Heroes set out"
        assert loaded_chapter["default_location"] == "village square"
        assert len(loaded_chapter["pages"]) >= 2

    def test_validate_after_full_setup(self, client):
        """After a full setup the story graph should be valid."""
        luna = create_character(client, "Luna")
        create_chapter(client, "Chapter 1", [luna["character_id"]])

        response = client.get("/api/story/validate")
        assert response.status_code == 200
        validation = response.json()
        assert validation["valid"] is True
        assert validation["errors"] == []


# ---------- Edge cases ----------


class TestEdgeCases:
    def test_chapter_requires_at_least_one_character(self, client):
        response = client.post("/api/chapters", json={
            "title": "Empty Chapter",
            "character_ids": [],
        })
        assert response.status_code == 400

    def test_nonexistent_character_returns_404(self, client):
        response = client.put("/api/characters/nonexistent", json={"name": "Ghost"})
        assert response.status_code == 404

    def test_nonexistent_chapter_returns_404(self, client):
        response = client.put("/api/chapters/nonexistent", json={"title": "Ghost"})
        assert response.status_code == 404

    def test_nonexistent_page_returns_404(self, client):
        response = client.put("/api/pages/nonexistent", json={"mood": "sad"})
        assert response.status_code == 404

    def test_nonexistent_panel_returns_404(self, client):
        response = client.put("/api/panels/nonexistent", json={"narration": "..."})
        assert response.status_code == 404

    def test_nonexistent_script_returns_404(self, client):
        response = client.put("/api/scripts/nonexistent", json={"dialogue": "..."})
        assert response.status_code == 404

    def test_new_story_endpoint(self, client):
        response = client.post("/api/story/new?title=Fresh%20Start")
        assert response.status_code == 200
        story_data = response.json()
        assert story_data["title"] == "Fresh Start"
        assert len(story_data["characters"]) == 0
        assert len(story_data["chapters"]) == 0
