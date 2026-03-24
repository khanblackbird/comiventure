/**
 * Comic page renderer — builds panel grids, speech bubbles, and narration.
 * Receives layout data from the backend ComicComposer.
 */
class ComicRenderer {
    constructor(containerElement) {
        this.container = containerElement;
        this.currentPageData = null;
        this.onPanelClick = null;
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

        // Image or video content
        if (panelData.is_animated && panelData.video_path) {
            const video = document.createElement('video');
            video.src = panelData.video_path;
            video.autoplay = true;
            video.loop = true;
            video.muted = true;
            panel.appendChild(video);
        } else if (panelData.image_path) {
            const img = document.createElement('img');
            img.src = panelData.image_path;
            img.alt = 'Comic panel';
            panel.appendChild(img);
        } else {
            const placeholder = document.createElement('div');
            placeholder.className = 'placeholder';
            placeholder.textContent = 'Panel awaiting generation...';
            panel.appendChild(placeholder);
        }

        // Narration box
        if (panelData.narration) {
            const narration = document.createElement('div');
            narration.className = 'narration-box';
            narration.textContent = panelData.narration;
            panel.appendChild(narration);
        }

        // Speech bubbles
        if (panelData.dialogue) {
            for (let bubbleIndex = 0; bubbleIndex < panelData.dialogue.length; bubbleIndex++) {
                const dialogue = panelData.dialogue[bubbleIndex];
                const bubble = this._createBubble(dialogue, bubbleIndex);
                panel.appendChild(bubble);
            }
        }

        // Click to edit
        panel.addEventListener('click', () => {
            if (this.onPanelClick) {
                this.onPanelClick(panelData);
            }
        });

        return panel;
    }

    _createBubble(dialogue, index) {
        const bubble = document.createElement('div');
        bubble.className = 'speech-bubble';
        bubble.style.top = `${20 + index * 60}px`;
        bubble.style.left = `${10 + (index % 2) * 40}%`;

        const speaker = document.createElement('div');
        speaker.className = 'speaker';
        speaker.textContent = dialogue.character;
        bubble.appendChild(speaker);

        const text = document.createElement('div');
        text.textContent = dialogue.text;
        bubble.appendChild(text);

        return bubble;
    }

    renderPlaceholderPage(panelCount = 4) {
        const placeholderData = {
            page_id: 'placeholder',
            template: panelCount <= 1 ? 'single' : panelCount <= 2 ? 'two_equal' : panelCount <= 3 ? 'hero_top' : 'grid_2x2',
            panels: Array.from({ length: panelCount }, (_, index) => ({
                panel_id: `placeholder-${index}`,
                image_path: null,
                video_path: null,
                is_animated: false,
                dialogue: [],
                narration: '',
            })),
        };
        this.renderPage(placeholderData);
    }
}
