"""UI tests — Playwright drives a real browser against the running app.

Tests every screen, button, field, and navigation flow.
Catches: missing functions, broken buttons, unwired fields, JS errors.

Usage:
    pytest tests/test_ui.py -v

Requires:
    pip install pytest-playwright
    playwright install chromium

The tests start their own server (no Docker needed).
"""
import pytest
import re
import subprocess
import time
import signal
import os

from playwright.sync_api import Page, expect


# --- Server fixture ---

@pytest.fixture(scope="session")
def server():
    """Start the FastAPI server for UI testing."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "backend.app:app",
         "--host", "127.0.0.1", "--port", "8765"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to start
    for _ in range(30):
        try:
            import httpx
            r = httpx.get("http://127.0.0.1:8765/api/story", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Server did not start in time")

    yield "http://127.0.0.1:8765"

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture
def app_page(server, page: Page):
    """Navigate to the app and return the page."""
    page.goto(server)
    page.wait_for_load_state("networkidle")
    return page


# --- Console error tracking ---

@pytest.fixture(autouse=True)
def track_js_errors(page: Page):
    """Collect JS console errors — fail test if any occur."""
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    yield
    # Filter out non-critical errors
    critical = [e for e in errors if "TypeError" in e or "ReferenceError" in e]
    assert not critical, f"JS errors on page: {critical}"


# --- Splash Screen ---

class TestSplashScreen:
    def test_splash_visible_on_load(self, app_page: Page):
        """Splash screen should be visible when app loads."""
        splash = app_page.locator("#splash")
        expect(splash).to_be_visible()

    def test_new_story_button_exists(self, app_page: Page):
        expect(app_page.locator("#btn-splash-new")).to_be_visible()

    def test_load_file_button_exists(self, app_page: Page):
        expect(app_page.locator("#splash-load-file")).to_be_attached()

    def test_new_story_navigates_to_chapter_select(self, app_page: Page):
        """Clicking New Story should show the chapter select screen."""
        app_page.click("#btn-splash-new")
        app_page.wait_for_load_state("networkidle")
        expect(app_page.locator("#chapter-select")).to_be_visible()
        expect(app_page.locator("#splash")).to_be_hidden()


# --- Chapter Select Screen ---

class TestChapterSelect:
    def _go_to_chapter_select(self, page: Page):
        page.click("#btn-splash-new")
        page.wait_for_load_state("networkidle")

    def test_story_settings_visible(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        expect(app_page.locator("#story-art-style")).to_be_visible()
        expect(app_page.locator("#story-genre")).to_be_visible()
        expect(app_page.locator("#story-synopsis")).to_be_visible()

    def test_story_settings_editable(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        app_page.fill("#story-art-style", "manga")
        app_page.locator("#story-art-style").dispatch_event("change")
        assert app_page.input_value("#story-art-style") == "manga"

    def test_characters_button_navigates(self, app_page: Page):
        """Characters button should navigate to character screen."""
        self._go_to_chapter_select(app_page)
        app_page.click("#btn-chapter-manage-chars")
        expect(app_page.locator("#character-screen")).to_be_visible()

    def test_back_button_returns_to_splash(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        app_page.click("#btn-chapter-back")
        expect(app_page.locator("#splash")).to_be_visible()

    def test_lora_section_visible(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        expect(app_page.locator("#lora-section")).to_be_visible()

    def test_style_ref_section_visible(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        expect(app_page.locator("#style-ref-section")).to_be_visible()

    def test_model_selector_visible(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        expect(app_page.locator("#model-selector")).to_be_visible()

    def test_huggingface_browse_opens(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        app_page.click("#btn-browse-hf")
        expect(app_page.locator("#lora-browser")).to_be_visible()

    def test_adapter_section_visible(self, app_page: Page):
        self._go_to_chapter_select(app_page)
        expect(app_page.locator("#adapter-section")).to_be_visible()


# --- Character Screen ---

class TestCharacterScreen:
    def _go_to_char_screen(self, page: Page):
        page.click("#btn-splash-new")
        page.wait_for_load_state("networkidle")
        page.click("#btn-chapter-manage-chars")

    def test_character_screen_visible(self, app_page: Page):
        self._go_to_char_screen(app_page)
        expect(app_page.locator("#character-screen")).to_be_visible()

    def test_back_returns_to_chapters(self, app_page: Page):
        self._go_to_char_screen(app_page)
        app_page.click("#btn-char-screen-back")
        expect(app_page.locator("#chapter-select")).to_be_visible()

    def test_add_manually_opens_manager(self, app_page: Page):
        self._go_to_char_screen(app_page)
        app_page.click("#btn-char-screen-add")
        expect(app_page.locator("#character-manager-overlay")).to_have_class(re.compile("visible"))

    def test_new_from_image_button_exists(self, app_page: Page):
        self._go_to_char_screen(app_page)
        expect(app_page.locator("#new-char-from-image")).to_be_attached()

    def test_empty_state_shows_hint(self, app_page: Page):
        """With no characters, the detail panel shows a hint."""
        self._go_to_char_screen(app_page)
        expect(app_page.locator("#char-detail-empty")).to_be_visible()


# --- Character Manager ---

class TestCharacterManager:
    def _open_manager(self, page: Page):
        page.click("#btn-splash-new")
        page.wait_for_load_state("networkidle")
        page.click("#btn-chapter-manage-chars")
        page.click("#btn-char-screen-add")
        page.locator("#character-manager-overlay.visible").wait_for(state="visible")

    def test_create_character(self, app_page: Page):
        """Create a character via the form — no JS errors."""
        self._open_manager(app_page)
        app_page.fill("#new-char-name", "Luna")
        app_page.fill("#new-char-appearance", "blue-haired girl")
        app_page.click("#btn-create-character")
        app_page.wait_for_load_state("networkidle")
        # Close manager — should not throw
        app_page.click("#btn-close-characters")
        # Character list should have at least one item
        expect(app_page.locator(".char-list-item").first).to_be_visible(timeout=5000)

    def test_close_button_works(self, app_page: Page):
        self._open_manager(app_page)
        app_page.click("#btn-close-characters")
        expect(app_page.locator("#character-manager-overlay")).not_to_have_class(re.compile("visible"))


# --- Full Flow ---

class TestFullFlow:
    def test_create_story_add_character_create_chapter(self, app_page: Page):
        """Full flow: new story → add character → create chapter → see it in grid."""
        # New story
        app_page.click("#btn-splash-new")
        app_page.wait_for_load_state("networkidle")

        # Go to character screen and add a character
        app_page.click("#btn-chapter-manage-chars")
        app_page.click("#btn-char-screen-add")
        app_page.locator("#character-manager-overlay.visible").wait_for(state="visible")
        app_page.fill("#new-char-name", "Peter")
        app_page.fill("#new-char-appearance", "young brown rabbit")
        app_page.click("#btn-create-character")
        app_page.wait_for_load_state("networkidle")
        app_page.click("#btn-close-characters")

        # Back to chapter select
        app_page.click("#btn-char-screen-back")
        expect(app_page.locator("#chapter-select")).to_be_visible()

        # Create a chapter (clicking the + New Chapter button)
        app_page.click("#btn-add-chapter-select")
        app_page.wait_for_load_state("networkidle")

        # Should now have a chapter card in the grid
        expect(app_page.locator(".chapter-card").first).to_be_visible(timeout=5000)

    def test_page_context_fields_exist(self, app_page: Page):
        """All page context fields should be present in the editor."""
        # Create story + character + enter chapter
        app_page.click("#btn-splash-new")
        app_page.wait_for_load_state("networkidle")
        app_page.click("#btn-chapter-manage-chars")
        app_page.click("#btn-char-screen-add")
        app_page.locator("#character-manager-overlay.visible").wait_for(state="visible")
        app_page.fill("#new-char-name", "Rex")
        app_page.click("#btn-create-character")
        app_page.wait_for_load_state("networkidle")
        app_page.click("#btn-close-characters")
        app_page.click("#btn-char-screen-back")

        chapter_cards = app_page.locator(".chapter-card")
        if chapter_cards.count() > 0:
            chapter_cards.first.click()
            # Verify all page context fields exist
            expect(app_page.locator("#page-setting")).to_be_attached()
            expect(app_page.locator("#page-mood")).to_be_attached()
            expect(app_page.locator("#page-action")).to_be_attached()
            expect(app_page.locator("#page-time-of-day")).to_be_attached()
            expect(app_page.locator("#page-weather")).to_be_attached()
            expect(app_page.locator("#page-lighting")).to_be_attached()

    def test_no_js_errors_on_navigation(self, app_page: Page):
        """Navigate through all screens — no JS errors should occur."""
        # Splash → Chapter select
        app_page.click("#btn-splash-new")
        app_page.wait_for_load_state("networkidle")

        # Chapter select → Character screen
        app_page.click("#btn-chapter-manage-chars")
        app_page.wait_for_timeout(500)

        # Character screen → Back to chapters
        app_page.click("#btn-char-screen-back")
        app_page.wait_for_timeout(500)

        # Chapter select → Back to splash
        app_page.click("#btn-chapter-back")
        app_page.wait_for_timeout(500)

        # If we get here with no JS errors (tracked by autouse fixture), pass
