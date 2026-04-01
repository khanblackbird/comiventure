"""Tests that the app starts up without import errors.

Every dependency must be importable. The app module must load.
The FastAPI app must be constructable and serve routes.

These catch missing packages (like python-multipart) that only
show up at import time when a route uses File/Form.
"""
import pytest
from fastapi.testclient import TestClient


class TestAppImports:
    """All modules must import without error."""

    def test_app_module_imports(self):
        from backend import app
        assert app.app is not None

    def test_routes_module_imports(self):
        from backend.api import routes
        assert routes.router is not None

    def test_models_import(self):
        from backend.models import (
            Emitter, ContentStore, ContentMeta,
            Appearance, AppearanceProperties, ReferenceImage,
            Script, Panel, Page, Chapter, Character, Story,
        )

    def test_generator_imports(self):
        from backend.generator import ImageGenerator, PanelGenerator

    def test_multipart_available(self):
        """python-multipart must be installed for file upload endpoints."""
        import python_multipart
        assert python_multipart is not None


class TestAppStartup:
    """The app must start and serve basic routes."""

    def test_app_creates_test_client(self):
        from backend.app import app
        client = TestClient(app)
        assert client is not None

    def test_root_returns_html(self):
        from backend.app import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "Comiventure" in response.text

    def test_story_endpoint_responds(self):
        from backend.app import app
        from backend.api import routes
        from backend.models import Story
        routes.story = Story("test", "Test")
        client = TestClient(app)
        response = client.get("/api/story")
        assert response.status_code == 200

    def test_characters_endpoint_responds(self):
        from backend.app import app
        from backend.api import routes
        from backend.models import Story
        routes.story = Story("test", "Test")
        client = TestClient(app)
        response = client.get("/api/characters")
        assert response.status_code == 200

    def test_validate_endpoint_responds(self):
        from backend.app import app
        from backend.api import routes
        from backend.models import Story, Character
        routes.story = Story("test", "Test")
        client = TestClient(app)
        response = client.get("/api/story/validate")
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "errors" in data
