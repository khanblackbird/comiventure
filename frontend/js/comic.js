/**
 * Comic page renderer — builds panel grids, speech bubbles, and narration.
 * Receives layout data from the backend ComicComposer.
 */
class ComicRenderer {
    constructor(containerElement) {
        this.container = containerElement;
        this.currentPageData = null;
        this.onPanelSelect = null;
        this.selectedPanelId = null;
    }

    renderPage(pageData) {
        this.currentPageData = pageData;
        this.container.innerHTML = '';
        this.container.className = '';

        const template = pageData.template || 'single';
        this.container.classList.add(`layout-${template}`);

        for (const panelData of pageData.panels) {
            const panelElement = this._createPanelElement(panelData);
            this.container.appendChild(panelElement);
        }
    }

    _createPanelElement(panelData) {
        const panel = document.createElement('div');
        panel.className = 'comic-panel';
        panel.dataset.panelId = panelData.panel_id;

        // Image or video content — supports both paths and content hashes
        const imageSource = panelData.image_url || panelData.image_path;
        if (panelData.is_animated && panelData.video_path) {
            const video = document.createElement('video');
            video.src = panelData.video_path;
            video.autoplay = true;
            video.loop = true;
            video.muted = true;
            panel.appendChild(video);
        } else if (imageSource) {
            const img = document.createElement('img');
            img.src = imageSource;
            img.alt = 'Comic panel';
            panel.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.className = 'placeholder';
            placeholder.textContent = 'Panel awaiting generation...';
            panel.appendChild(placeholder);
        }

        // Dialogue bar at the bottom
        const hasNarration = panelData.narration && panelData.narration.trim();
        const hasDialogue = panelData.dialogue && panelData.dialogue.length > 0;

        if (hasNarration || hasDialogue) {
            const dialogueBar = document.createElement('div');
            dialogueBar.className = 'panel-dialogue';

            if (hasNarration) {
                const narration = document.createElement('div');
                narration.className = 'narration-box';
                narration.textContent = panelData.narration;
                dialogueBar.appendChild(narration);
            }

            if (hasDialogue) {
                for (const dialogue of panelData.dialogue) {
                    const bubble = this._createBubble(dialogue);
                    dialogueBar.appendChild(bubble);
                }
            }

            panel.appendChild(dialogueBar);
        }

        // Click to select
        panel.addEventListener('click', () => {
            this._selectPanel(panelData.panel_id);
            if (this.onPanelSelect) {
                this.onPanelSelect(panelData);
            }
        });

        return panel;
    }

    _createBubble(dialogue) {
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';

        const speaker = document.createElement('span');
        speaker.className = 'speaker';
        speaker.textContent = dialogue.character + ':';
        bubble.appendChild(speaker);

        const text = document.createTextNode(' ' + dialogue.text);
        bubble.appendChild(text);

        return bubble;
    }

    _selectPanel(panelId) {
        this.selectedPanelId = panelId;
        const allPanels = this.container.querySelectorAll('.comic-panel');
        for (const panel of allPanels) {
            panel.classList.toggle('selected', panel.dataset.panelId === panelId);
        }
    }

    getSelectedPanelData() {
        if (!this.selectedPanelId || !this.currentPageData) return null;
        return this.currentPageData.panels.find(
            panel => panel.panel_id === this.selectedPanelId
        ) || null;
    }

    renderPlaceholderPage(panelCount = 4) {
        const placeholderData = {
            page_id: 'placeholder',
            template: panelCount <= 1 ? 'single' : panelCount <= 2 ? 'two_equal' : panelCount <= 3 ? 'hero_top' : 'grid_2x2',
            panels: Array.from({ length: panelCount }, (_, index) => ({
                panel_id: `placeholder-${index}`,
                image_path: null,
                image_url: null,
                video_path: null,
                is_animated: false,
                dialogue: [],
                narration: '',
                scene_prompt: '',
                character_ids: [],
            })),
        };
        this.renderPage(placeholderData);
    }
}
