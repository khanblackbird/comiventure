/**
 * API client — all calls go through the hierarchy-enforcing backend.
 */
class ComiventureAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async _post(path, body) {
        const response = await fetch(`${this.baseUrl}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `${response.status} ${response.statusText}`);
        }
        return response.json();
    }

    async _put(path, body) {
        const response = await fetch(`${this.baseUrl}${path}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `${response.status} ${response.statusText}`);
        }
        return response.json();
    }

    async _get(path) {
        const response = await fetch(`${this.baseUrl}${path}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `${response.status} ${response.statusText}`);
        }
        return response.json();
    }

    async _delete(path) {
        const response = await fetch(`${this.baseUrl}${path}`, { method: 'DELETE' });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `${response.status} ${response.statusText}`);
        }
        return response.json();
    }

    // Generic PUT shorthand
    put(path, body) { return this._put(path, body); }

    // Story
    getStory() { return this._get('/api/story'); }
    newStory(title) { return this._post('/api/story/new', { title }); }
    updateStory(data) { return this._put('/api/story', data); }
    saveStory() { return this._post('/api/story/save', {}); }
    listSavedStories() { return this._get('/api/stories'); }
    loadSavedStory(filename) { return this._post(`/api/story/load/${filename}`, {}); }

    async uploadStory(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/story/load', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(await response.text());
        return response.json();
    }

    async importCharacter(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/story/import-character', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(await response.text());
        return response.json();
    }

    downloadStoryUrl() { return '/api/story/download'; }

    // Characters
    listCharacters() { return this._get('/api/characters'); }
    createCharacter(data) { return this._post('/api/characters', data); }
    updateCharacter(characterId, data) { return this._put(`/api/characters/${characterId}`, data); }
    deleteCharacter(characterId) { return this._delete(`/api/characters/${characterId}`); }

    // Chapters
    listChapters() { return this._get('/api/chapters'); }
    createChapter(data) { return this._post('/api/chapters', data); }
    updateChapter(chapterId, data) { return this._put(`/api/chapters/${chapterId}`, data); }

    // Chapter membership
    addCharacterToChapter(chapterId, characterId) {
        return this._post(`/api/chapters/${chapterId}/characters/${characterId}`, {});
    }

    // Pages
    createPage(chapterId) { return this._post('/api/pages', { chapter_id: chapterId }); }

    // Panels
    createPanel(pageId) { return this._post('/api/panels', { page_id: pageId }); }
    updatePanel(panelId, data) { return this._put(`/api/panels/${panelId}`, data); }

    // Scripts
    createScript(data) { return this._post('/api/scripts', data); }
    updateScript(scriptId, data) { return this._put(`/api/scripts/${scriptId}`, data); }

    // Generation
    generatePanel(panelId, options = {}) {
        return this._post('/api/generate', { panel_id: panelId, ...options });
    }

    // Inpainting
    inpaintPanel(panelId, maskDataBase64, prompt, options = {}) {
        return this._post('/api/inpaint', {
            panel_id: panelId,
            mask_data: maskDataBase64,
            prompt: prompt,
            negative_prompt: options.negative_prompt || null,
            strength: options.strength || 0.75,
            steps: options.steps || 25,
            seed: options.seed || null,
        });
    }

    // Character chat
    chatWithCharacter(characterId, message, panelId = null, history = null) {
        return this._post('/api/chat/character', {
            character_id: characterId,
            message: message,
            panel_id: panelId,
            history: history,
        });
    }

    characterReact(characterId, panelId) {
        return this._post('/api/chat/react', {
            character_id: characterId,
            panel_id: panelId,
        });
    }

    suggestScripts(characterId, panelId) {
        return this._post('/api/chat/suggest-scripts', {
            character_id: characterId,
            panel_id: panelId,
        });
    }

    // Profile
    getProfile(characterId) { return this._get(`/api/characters/${characterId}/profile`); }
    updateProfile(characterId, data) { return this._put(`/api/characters/${characterId}/profile`, data); }
    addOutfit(characterId, data) { return this._post(`/api/characters/${characterId}/outfits`, data); }

    // Page
    updatePage(pageId, data) { return this._put(`/api/pages/${pageId}`, data); }

    // Feedback
    submitFeedback(contentHash, prompt, accepted, characterIds = [], panelId = '') {
        return this._post('/api/feedback', {
            content_hash: contentHash,
            prompt: prompt,
            accepted: accepted,
            character_ids: characterIds,
            panel_id: panelId,
        });
    }
    getFeedback() { return this._get('/api/feedback'); }
    trainAdapter() { return this._post('/api/adapter/train', {}); }

    // Content
    contentUrl(contentHash) { return `/api/content/${contentHash}`; }
}

const api = new ComiventureAPI();
