"""Tests for character_chat.py — LLM-powered in-character conversation.

All tests mock httpx so no Ollama instance is needed.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.models import Character, Panel, Script
from backend.models.page import Page
from backend.generator.character_chat import CharacterChat


def _make_character():
    char = Character("c1", "Luna", description="A fierce warrior princess")
    char.personality_prompt = "brave and impulsive"
    return char


def _make_panel_with_script(character_id="c1"):
    panel = Panel("pan-1")
    script = Script("scr-1", character_id)
    script.pose = "standing"
    script.emotion = "smile"
    panel.add_script(script)
    return panel


def _mock_ollama_response(content: str):
    """Build a mock httpx response for Ollama /api/chat."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": content}
    }
    return mock_response


def _patch_httpx(mock_response):
    """Context manager that patches httpx.AsyncClient to return mock_response on post."""
    patcher = patch("httpx.AsyncClient")

    def setup(mock_client):
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        return mock_instance

    return patcher, setup


class TestSuggestScripts:
    """suggest_scripts parses LLM output into structured script fields."""

    @pytest.mark.asyncio
    async def test_parses_valid_response(self):
        response_text = (
            "pose: standing\n"
            "action: fighting\n"
            "emotion: angry\n"
            "outfit: armor\n"
            "direction: upper_body\n"
            "shot_type: medium\n"
            "narration: Luna charges into battle"
        )
        mock_resp = _mock_ollama_response(response_text)

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            chat = CharacterChat(ollama_host="http://fake:11434")
            char = _make_character()
            panel = _make_panel_with_script()

            result = await chat.suggest_scripts(char, panel)

            assert result["pose"] == "standing"
            assert result["action"] == "fighting"
            assert result["emotion"] == "angry"
            assert result["outfit"] == "armor"
            assert result["narration"] == "Luna charges into battle"

    @pytest.mark.asyncio
    async def test_normalizes_through_tag_vocabulary(self):
        """Visual fields should be normalized through find_closest_tag."""
        response_text = (
            "pose: Standing Up\n"
            "action: sword fighting\n"
            "emotion: very angry\n"
            "outfit: full plate armor\n"
            "direction: close up shot\n"
            "shot_type: wide\n"
            "narration: A fierce duel"
        )
        mock_resp = _mock_ollama_response(response_text)

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            chat = CharacterChat(ollama_host="http://fake:11434")
            char = _make_character()
            panel = _make_panel_with_script()

            result = await chat.suggest_scripts(char, panel)

            # Visual fields should be normalized (no spaces, lowercase)
            assert " " not in result["pose"] or result["pose"] == ""
            # narration is NOT normalized through tag vocab
            assert result["narration"] == "A fierce duel"
            # shot_type is NOT normalized through tag vocab
            assert result["shot_type"] == "wide"

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Empty LLM response should return empty fields gracefully."""
        mock_resp = _mock_ollama_response("")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            chat = CharacterChat(ollama_host="http://fake:11434")
            char = _make_character()
            panel = _make_panel_with_script()

            result = await chat.suggest_scripts(char, panel)

            assert isinstance(result, dict)
            assert all(result[k] == "" for k in result)

    @pytest.mark.asyncio
    async def test_handles_malformed_response(self):
        """Missing fields in LLM output should leave those fields empty."""
        response_text = (
            "pose: kneeling\n"
            "This is some random text the LLM added\n"
            "emotion: sad\n"
        )
        mock_resp = _mock_ollama_response(response_text)

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            chat = CharacterChat(ollama_host="http://fake:11434")
            char = _make_character()
            panel = _make_panel_with_script()

            result = await chat.suggest_scripts(char, panel)

            assert result["pose"] != ""
            assert result["emotion"] != ""
            # Fields not in the response remain empty
            assert result["action"] == ""
            assert result["outfit"] == ""
            assert result["narration"] == ""


class TestBuildSystemPrompt:
    """_build_system_prompt assembles context for the LLM."""

    def test_includes_character_profile(self):
        chat = CharacterChat(ollama_host="http://fake:11434")
        char = _make_character()

        prompt = chat._build_system_prompt(char)

        assert "Luna" in prompt
        assert "fierce warrior princess" in prompt
        assert "brave and impulsive" in prompt

    def test_includes_page_setting(self):
        chat = CharacterChat(ollama_host="http://fake:11434")
        char = _make_character()
        page = Page("pg-1", setting="dark forest", mood="tense")

        prompt = chat._build_system_prompt(char, page=page)

        assert "dark forest" in prompt
        assert "tense" in prompt

    def test_handles_none_page_and_panel(self):
        chat = CharacterChat(ollama_host="http://fake:11434")
        char = _make_character()

        prompt = chat._build_system_prompt(char, panel=None, page=None)

        # Should still produce a valid prompt with character info
        assert "Luna" in prompt
        assert isinstance(prompt, str)

    def test_includes_panel_narration(self):
        chat = CharacterChat(ollama_host="http://fake:11434")
        char = _make_character()
        panel = _make_panel_with_script()
        panel.narration = "The battle rages on"

        prompt = chat._build_system_prompt(char, panel=panel)

        assert "The battle rages on" in prompt

    def test_includes_page_action_context(self):
        chat = CharacterChat(ollama_host="http://fake:11434")
        char = _make_character()
        page = Page("pg-1", action_context="combat")

        prompt = chat._build_system_prompt(char, page=page)

        assert "combat" in prompt


class TestChatFallback:
    """Chat must gracefully handle failures."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_fallback(self):
        """When httpx raises, chat returns a fallback string."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = Exception("Connection refused")
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            chat = CharacterChat(ollama_host="http://fake:11434")
            char = _make_character()

            result = await chat.chat(char, "Hello!")

            assert "doesn't respond" in result
            assert "Luna" in result
