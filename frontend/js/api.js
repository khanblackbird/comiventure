/**
 * API client for communicating with the Comiventure backend.
 * All backend calls go through this module.
 */
class ComiventureAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.websocket = null;
        this.onProgressUpdate = null;
    }

    async getStory() {
        const response = await fetch(`${this.baseUrl}/api/story`);
        return response.json();
    }

    async getPages() {
        const response = await fetch(`${this.baseUrl}/api/pages`);
        return response.json();
    }

    async getPage(pageNumber) {
        const response = await fetch(`${this.baseUrl}/api/pages/${pageNumber}`);
        return response.json();
    }

    async sendChat(characterId, message) {
        const response = await fetch(`${this.baseUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character_id: characterId, message: message }),
        });
        return response.json();
    }

    async editPanel(panelId, maskDataBase64, prompt) {
        const response = await fetch(`${this.baseUrl}/api/edit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panel_id: panelId, mask_data: maskDataBase64, prompt: prompt }),
        });
        return response.json();
    }

    async animatePanel(panelId) {
        const response = await fetch(`${this.baseUrl}/api/animate/${panelId}`, {
            method: 'POST',
        });
        return response.json();
    }

    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws`;
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (this.onProgressUpdate) {
                this.onProgressUpdate(data);
            }
        };

        this.websocket.onclose = () => {
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
}

const api = new ComiventureAPI();
