"""Integration tests — hit the API then validate graph integrity.

These tests caught the critical bug where API routes created panels
without scripts, bypassing the model's cascade. Every API operation
that creates objects must leave the graph valid.
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


def _validate(client):
    """Hit the validate endpoint and assert no errors."""
    response = client.get("/api/story/validate")
    data = response.json()
    assert data["valid"], f"Graph integrity violated: {data['errors']}"
    return data


class TestCreateChapterIntegrity:
    def test_chapter_creation_leaves_valid_graph(self, client):
        client.post("/api/characters", json={"name": "Luna"})
        chars = client.get("/api/characters").json()
        char_id = list(chars.keys())[0]

        client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char_id],
        })

        _validate(client)

    def test_chapter_with_multiple_characters_valid(self, client):
        c1 = client.post("/api/characters", json={"name": "Luna"}).json()
        c2 = client.post("/api/characters", json={"name": "Rex"}).json()

        client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [c1["character_id"], c2["character_id"]],
        })

        data = _validate(client)

    def test_chapter_cascade_creates_page_panel_scripts(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        # Chapter must have pages
        assert len(chapter["pages"]) >= 1
        # Page must have panels
        page = chapter["pages"][0]
        assert len(page["panels"]) >= 1
        # Panel must have scripts
        panel = page["panels"][0]
        assert len(panel["scripts"]) >= 1
        # Script must reference the character
        assert char["character_id"] in panel["scripts"]

        _validate(client)


class TestCreatePageIntegrity:
    def test_page_creation_leaves_valid_graph(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        page = client.post("/api/pages", json={
            "chapter_id": chapter["chapter_id"],
        }).json()

        # Page must have panel with scripts
        assert len(page["panels"]) >= 1
        panel = page["panels"][0]
        assert len(panel["scripts"]) >= 1
        assert char["character_id"] in panel["scripts"]

        _validate(client)

    def test_page_panel_has_all_chapter_characters(self, client):
        c1 = client.post("/api/characters", json={"name": "Luna"}).json()
        c2 = client.post("/api/characters", json={"name": "Rex"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [c1["character_id"], c2["character_id"]],
        }).json()

        page = client.post("/api/pages", json={
            "chapter_id": chapter["chapter_id"],
        }).json()

        panel = page["panels"][0]
        assert c1["character_id"] in panel["scripts"]
        assert c2["character_id"] in panel["scripts"]

        _validate(client)


class TestCreatePanelIntegrity:
    def test_panel_creation_leaves_valid_graph(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        page_id = chapter["pages"][0]["page_id"]
        panel_response = client.post("/api/panels", json={"page_id": page_id})
        assert panel_response.status_code == 200

        panel = panel_response.json()
        assert len(panel["scripts"]) >= 1
        assert char["character_id"] in panel["scripts"]

        _validate(client)

    def test_panel_has_all_chapter_characters(self, client):
        c1 = client.post("/api/characters", json={"name": "Luna"}).json()
        c2 = client.post("/api/characters", json={"name": "Rex"}).json()
        c3 = client.post("/api/characters", json={"name": "Kai"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [c1["character_id"], c2["character_id"], c3["character_id"]],
        }).json()

        page_id = chapter["pages"][0]["page_id"]
        panel = client.post("/api/panels", json={"page_id": page_id}).json()

        assert c1["character_id"] in panel["scripts"]
        assert c2["character_id"] in panel["scripts"]
        assert c3["character_id"] in panel["scripts"]

        _validate(client)


class TestScriptIntegrity:
    def test_adding_script_leaves_valid_graph(self, client):
        c1 = client.post("/api/characters", json={"name": "Luna"}).json()
        c2 = client.post("/api/characters", json={"name": "Rex"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [c1["character_id"], c2["character_id"]],
        }).json()

        panel = chapter["pages"][0]["panels"][0]
        # Scripts already exist from cascade — update one
        script_id = panel["scripts"][c1["character_id"]]["script_id"]
        client.put(f"/api/scripts/{script_id}", json={
            "dialogue": "Hello!",
            "action": "waves",
        })

        _validate(client)


class TestFullWorkflow:
    """Simulate the actual user flow and validate at every step."""

    def test_full_story_creation_workflow(self, client):
        # Create characters — graph is incomplete until chapter exists
        c1 = client.post("/api/characters", json={
            "name": "Luna",
            "appearance_prompt": "blue-haired anime girl",
        }).json()
        c2 = client.post("/api/characters", json={
            "name": "Rex",
            "appearance_prompt": "tall wolf anthro",
        }).json()
        # Don't validate yet — characters without chapters is intermediate state

        # Create chapter — now the graph should be complete
        chapter = client.post("/api/chapters", json={
            "title": "The Meeting",
            "character_ids": [c1["character_id"], c2["character_id"]],
        }).json()
        _validate(client)

        # Add another page
        page = client.post("/api/pages", json={
            "chapter_id": chapter["chapter_id"],
        }).json()
        _validate(client)

        # Add another panel
        panel = client.post("/api/panels", json={
            "page_id": page["page_id"],
        }).json()
        _validate(client)

        # Update scripts
        for char_id, script_data in panel["scripts"].items():
            client.put(f"/api/scripts/{script_data['script_id']}", json={
                "dialogue": f"Hello from {char_id}",
                "action": "waves",
                "emotion": "happy",
            })
        _validate(client)

        # Final story state check
        story = client.get("/api/story").json()
        assert len(story["characters"]) == 2
        assert len(story["chapters"]) == 1
        chapter_data = list(story["chapters"].values())[0]
        # cascade page + manually created page
        assert len(chapter_data["pages"]) >= 2
        for page_data in chapter_data["pages"]:
            assert len(page_data["panels"]) >= 1
            for panel_data in page_data["panels"]:
                assert len(panel_data["scripts"]) >= 1


class TestNoOrphanObjects:
    """No API operation should create orphaned objects."""

    def test_no_scriptless_panels_after_chapter_create(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        for page in chapter["pages"]:
            for panel in page["panels"]:
                assert len(panel["scripts"]) > 0, \
                    f"Panel {panel['panel_id']} has no scripts — orphaned!"

    def test_no_scriptless_panels_after_page_create(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        page = client.post("/api/pages", json={
            "chapter_id": chapter["chapter_id"],
        }).json()

        for panel in page["panels"]:
            assert len(panel["scripts"]) > 0, \
                f"Panel {panel['panel_id']} has no scripts — orphaned!"

    def test_no_scriptless_panels_after_panel_create(self, client):
        char = client.post("/api/characters", json={"name": "Luna"}).json()
        chapter = client.post("/api/chapters", json={
            "title": "Ch1",
            "character_ids": [char["character_id"]],
        }).json()

        page_id = chapter["pages"][0]["page_id"]
        panel = client.post("/api/panels", json={"page_id": page_id}).json()

        assert len(panel["scripts"]) > 0, \
            f"Panel {panel['panel_id']} has no scripts — orphaned!"
