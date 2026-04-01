"""Tests that the API enforces the hierarchy.

Every API operation that creates objects must use factory methods
and leave the graph valid.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.api import routes
from backend.models import Story, ContentStore


@pytest.fixture(autouse=True)
def fresh_story(tmp_path):
    routes.story = Story("test", "Test")
    routes.content_store = ContentStore(str(tmp_path / "content"))
    routes.image_generator = None
    yield
    routes.story = None


@pytest.fixture
def client():
    return TestClient(app)


class TestHierarchyEnforcement:
    def test_cannot_create_chapter_without_characters(self, client):
        response = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [],
        })
        assert response.status_code == 400

    def test_cannot_create_chapter_with_nonexistent_character(self, client):
        response = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": ["fake"],
        })
        assert response.status_code == 404

    def test_cannot_create_page_without_chapter(self, client):
        response = client.post("/api/pages", json={"chapter_id": "fake"})
        assert response.status_code == 404

    def test_cannot_create_panel_without_page(self, client):
        response = client.post("/api/panels", json={"page_id": "fake"})
        assert response.status_code == 404

    def test_cannot_create_script_without_panel(self, client):
        response = client.post("/api/scripts", json={
            "panel_id": "fake", "character_id": "fake",
        })
        assert response.status_code == 404

    def test_cannot_create_script_with_wrong_character(self, client):
        luna = client.post("/api/characters", json={"name": "Luna"}).json()
        rex = client.post("/api/characters", json={"name": "Rex"}).json()

        chapter = client.post("/api/chapters", json={
            "title": "Luna Only", "character_ids": [luna["character_id"]],
        }).json()

        panel_id = chapter["pages"][0]["panels"][0]["panel_id"]

        response = client.post("/api/scripts", json={
            "panel_id": panel_id,
            "character_id": rex["character_id"],
            "dialogue": "I shouldn't be here",
        })
        assert response.status_code == 400
        assert "not in this panel's chapter" in response.json()["detail"]

    def test_cannot_generate_without_panel(self, client):
        response = client.post("/api/generate", json={"panel_id": "fake"})
        assert response.status_code == 404


class TestChapterCascade:
    def test_chapter_creates_page_panel_scripts(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [char["character_id"]],
        }).json()

        assert len(chapter["pages"]) >= 1
        page = chapter["pages"][0]
        assert len(page["panels"]) >= 1
        panel = page["panels"][0]
        assert len(panel["scripts"]) >= 1
        assert char["character_id"] in panel["scripts"]

    def test_multi_character_scripts(self, client):
        c1 = client.post("/api/characters", json={"name": "Luna"}).json()
        c2 = client.post("/api/characters", json={"name": "Rex"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [c1["character_id"], c2["character_id"]],
        }).json()

        panel = chapter["pages"][0]["panels"][0]
        assert c1["character_id"] in panel["scripts"]
        assert c2["character_id"] in panel["scripts"]


class TestPageCascade:
    def test_page_creates_panel_with_scripts(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [char["character_id"]],
        }).json()

        page = client.post("/api/pages", json={
            "chapter_id": chapter["chapter_id"],
        }).json()

        assert len(page["panels"]) >= 1
        assert char["character_id"] in page["panels"][0]["scripts"]


class TestPanelCascade:
    def test_panel_creates_scripts(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [char["character_id"]],
        }).json()

        page_id = chapter["pages"][0]["page_id"]
        panel = client.post("/api/panels", json={"page_id": page_id}).json()

        assert len(panel["scripts"]) >= 1
        assert char["character_id"] in panel["scripts"]


class TestScriptUpdate:
    def test_update_cascade_script(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [char["character_id"]],
        }).json()

        script_id = chapter["pages"][0]["panels"][0]["scripts"][char["character_id"]]["script_id"]
        updated = client.put(f"/api/scripts/{script_id}", json={
            "dialogue": "Hello!",
            "action": "waves",
            "emotion": "happy",
        }).json()

        assert updated["dialogue"] == "Hello!"
        assert updated["source"] == "manual"


class TestStoryState:
    def test_full_story_reflects_hierarchy(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        client.post("/api/chapters", json={
            "title": "Ch1", "character_ids": [char["character_id"]],
        })

        story_data = client.get("/api/story").json()
        assert len(story_data["characters"]) == 1
        assert len(story_data["chapters"]) == 1
        chapter = list(story_data["chapters"].values())[0]
        assert len(chapter["pages"]) >= 1
        assert len(chapter["pages"][0]["panels"]) >= 1
        assert len(chapter["pages"][0]["panels"][0]["scripts"]) >= 1
