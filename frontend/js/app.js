/**
 * Main application — wires together all UI components and the API client.
 */
class ComiventureApp {
    constructor() {
        this.comicRenderer = new ComicRenderer(document.getElementById('page-container'));
        this.chatController = new ChatController(
            document.getElementById('chat-history'),
            document.getElementById('chat-input'),
            document.getElementById('btn-send'),
            document.getElementById('character-select'),
            document.getElementById('choice-area'),
        );
        this.panelEditor = new PanelEditor(
            document.getElementById('edit-overlay'),
            document.getElementById('edit-canvas'),
            document.getElementById('edit-prompt'),
            document.getElementById('brush-size'),
        );

        this.currentPage = 1;
        this._bindEvents();
        this._init();
    }

    _bindEvents() {
        // Chat sends message to backend
        this.chatController.onSendMessage = async (characterId, message) => {
            const response = await api.sendChat(characterId, message);
            if (response.character_response) {
                this.chatController.addMessage('character', response.character_name, response.character_response);
            }
            if (response.new_page) {
                this.comicRenderer.renderPage(response.new_page);
            }
            if (response.choices) {
                this.chatController.showChoices(response.choices);
            }
        };

        // Panel click opens editor
        this.comicRenderer.onPanelClick = (panelData) => {
            this.panelEditor.open(panelData);
        };

        // Editor applies edit via backend
        this.panelEditor.onApplyEdit = async (panelId, maskBase64, prompt) => {
            const response = await api.editPanel(panelId, maskBase64, prompt);
            if (response.updated_page) {
                this.comicRenderer.renderPage(response.updated_page);
            }
        };

        // Page navigation
        document.getElementById('btn-prev-page').addEventListener('click', () => this._navigatePage(-1));
        document.getElementById('btn-next-page').addEventListener('click', () => this._navigatePage(1));

        // Header buttons
        document.getElementById('btn-new-story').addEventListener('click', () => this._newStory());
        document.getElementById('btn-save').addEventListener('click', () => this._saveStory());
        document.getElementById('btn-load').addEventListener('click', () => this._loadStory());
    }

    async _init() {
        // Show placeholder UI while backend services are not yet connected
        this.comicRenderer.renderPlaceholderPage(4);

        this.chatController.renderCharacters([
            { character_id: 'demo-1', name: 'Luna', portrait_path: null },
            { character_id: 'demo-2', name: 'Rex', portrait_path: null },
        ]);

        this.chatController.addMessage('character', 'System', 'Welcome to Comiventure! Select a character and start chatting to build your story.');

        // Connect WebSocket for progress updates
        api.onProgressUpdate = (data) => this._handleProgress(data);
        // api.connectWebSocket();  // Uncomment when backend is running
    }

    _handleProgress(data) {
        const progressBar = document.getElementById('progress-bar');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        if (data.status === 'generating') {
            progressBar.hidden = false;
            progressFill.style.width = `${data.progress || 0}%`;
            progressText.textContent = data.message || 'Generating...';
        } else if (data.status === 'complete') {
            progressBar.hidden = true;
        }
    }

    async _navigatePage(direction) {
        this.currentPage += direction;
        const pageData = await api.getPage(this.currentPage);
        if (pageData.panels) {
            this.comicRenderer.renderPage(pageData);
        }
        document.getElementById('page-indicator').textContent = `Page ${this.currentPage}`;
        document.getElementById('btn-prev-page').disabled = this.currentPage <= 1;
    }

    async _newStory() {
        // TODO: reset story state
        this.comicRenderer.renderPlaceholderPage(4);
        this.currentPage = 1;
    }

    async _saveStory() {
        // TODO: persist story to local storage or file
    }

    async _loadStory() {
        // TODO: load story from local storage or file
    }
}

// Boot the app
document.addEventListener('DOMContentLoaded', () => {
    window.comiventure = new ComiventureApp();
});
