/**
 * Main application — enforces the hierarchy through the API.
 *
 * Character -> Chapter -> Page -> Panel -> Script
 *
 * The UI is contextual:
 * - Sidebar shows chapter characters by default
 * - Selecting a panel shows its per-character scripts
 * - Generation requires a complete chain
 */
class ComiventureApp {
    constructor() {
        this.comicRenderer = new ComicRenderer(document.getElementById('page-container'));
        this.panelEditor = new PanelEditor();

        // State — mirrors the backend hierarchy
        this.characters = {};       // character_id -> character data
        this.chapters = [];         // ordered list of chapter data
        this.currentChapterIndex = 0;
        this.currentPageIndex = 0;
        this.selectedPanelId = null;
        this.isGenerating = false;
        this.defaultNegativePrompt = 'lowres, (worst quality, bad quality:1.2), bad anatomy, sketch, jpeg artefacts, signature, watermark, old, oldest, censored, bar_censor, simple background';

        this._bindEvents();
        this._init();
    }

    // --- Getters for current position in hierarchy ---

    get currentChapter() {
        return this.chapters[this.currentChapterIndex] || null;
    }

    get currentPages() {
        return this.currentChapter ? this.currentChapter.pages : [];
    }

    get currentPage() {
        return this.currentPages[this.currentPageIndex] || null;
    }

    get selectedPanel() {
        if (!this.selectedPanelId || !this.currentPage) return null;
        return this.currentPage.panels.find(panel => panel.panel_id === this.selectedPanelId) || null;
    }

    // --- Initialization ---

    async _init() {
        // Show splash screen first
        this._showSplash();
    }

    async _enterStory() {
        try {
            const storyData = await api.getStory();
            if (Object.keys(storyData.characters).length === 0) {
                await this._bootstrapStory();
            } else {
                this._loadStoryData(storyData);
            }
        } catch (error) {
            console.warn('Backend not ready:', error);
            await this._bootstrapStory();
        }

        this._showChapterSelect();
    }

    _showChapterSelect() {
        document.getElementById('splash').classList.add('hidden');
        document.getElementById('app').classList.add('hidden');
        document.getElementById('character-screen').classList.add('hidden');
        document.getElementById('chapter-select').classList.remove('hidden');

        document.getElementById('chapter-select-title').textContent = this.chapters.length > 0
            ? (this.chapters[0]?.title ? 'Select Chapter' : 'Select Chapter')
            : 'Select Chapter';

        this._renderChapterGrid();
        this._renderModelSelector();
        this._renderAdapterStatus();
        this._loadStorySettings();
        this._renderLoraSection();
        this._renderStyleReferences();
    }

    _updateModelIndicators(modelName) {
        const indicators = [
            document.getElementById('char-screen-model'),
            document.getElementById('app-model-indicator'),
        ];
        for (const el of indicators) {
            if (el) el.textContent = modelName ? `Model: ${modelName}` : '';
        }
    }

    async _renderModelSelector() {
        const container = document.getElementById('model-options');
        container.innerHTML = '';

        try {
            const data = await api._get('/api/models');
            const models = data.models;
            const currentModelId = data.current;

            // Update model indicators everywhere
            const currentModel = Object.values(models).find(m => m.id === currentModelId);
            this._updateModelIndicators(currentModel?.name || currentModelId);

            for (const [key, model] of Object.entries(models)) {
                const option = document.createElement('div');
                option.className = 'model-option';
                if (model.id === currentModelId) option.classList.add('active');

                const info = document.createElement('div');
                info.className = 'model-option-info';

                const name = document.createElement('div');
                name.className = 'model-option-name';
                name.textContent = model.name;
                info.appendChild(name);

                const desc = document.createElement('div');
                desc.className = 'model-option-desc';
                desc.textContent = model.description;
                info.appendChild(desc);

                const tags = document.createElement('div');
                tags.className = 'model-option-tags';
                for (const tag of model.tags || []) {
                    const tagEl = document.createElement('span');
                    tagEl.className = 'model-tag';
                    tagEl.textContent = tag;
                    tags.appendChild(tagEl);
                }
                info.appendChild(tags);

                option.appendChild(info);

                if (model.id === currentModelId) {
                    const status = document.createElement('span');
                    status.className = 'model-option-status';
                    status.textContent = 'Active';
                    option.appendChild(status);
                }

                option.addEventListener('click', () => this._switchModel(key, option));
                container.appendChild(option);
            }
        } catch (error) {
            container.innerHTML = '<span class="saved-story-meta">Could not load models</span>';
        }
    }

    async _switchModel(modelKey, optionElement) {
        if (optionElement.classList.contains('active')) return;

        const statusEl = document.createElement('span');
        statusEl.className = 'model-option-status';
        statusEl.textContent = 'Loading...';
        optionElement.appendChild(statusEl);

        try {
            await api._post(`/api/models/${modelKey}`, {});
            await this._renderModelSelector();
        } catch (error) {
            statusEl.textContent = 'Failed';
            console.error('Failed to switch model:', error);
        }
    }

    // --- LoRA Library ---

    async _renderLoraSection() {
        const container = document.getElementById('lora-active-list');
        container.innerHTML = '';

        try {
            const data = await api._get('/api/loras');
            const library = data.loras || [];
            const activeLoras = this.storyData?.style_loras || [];

            if (library.length === 0 && activeLoras.length === 0) {
                container.innerHTML = '<span class="hint">No LoRAs — upload or browse Civitai</span>';
                return;
            }

            // Show available LoRAs with toggle + strength
            for (const lora of library) {
                const active = activeLoras.find(l => l.name === lora.name);
                const row = document.createElement('div');
                row.className = 'lora-row';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.checked = !!active;
                row.appendChild(checkbox);

                const name = document.createElement('span');
                name.className = 'lora-name';
                name.textContent = `${lora.name} (${lora.size_mb}MB)`;
                row.appendChild(name);

                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = '0';
                slider.max = '100';
                slider.value = active ? Math.round(active.strength * 100) : 70;
                slider.className = 'lora-strength';
                row.appendChild(slider);

                const label = document.createElement('span');
                label.className = 'lora-strength-label';
                label.textContent = `${slider.value}%`;
                slider.addEventListener('input', () => {
                    label.textContent = `${slider.value}%`;
                });
                row.appendChild(label);

                const save = () => this._saveActiveLoras();
                checkbox.addEventListener('change', save);
                slider.addEventListener('change', save);

                row.dataset.loraName = lora.name;
                row.dataset.loraFilename = lora.filename;
                container.appendChild(row);
            }
        } catch (error) {
            container.innerHTML = '<span class="hint">Could not load LoRAs</span>';
        }
    }

    async _saveActiveLoras() {
        const rows = document.querySelectorAll('.lora-row');
        const loras = [];
        for (const row of rows) {
            const checkbox = row.querySelector('input[type="checkbox"]');
            const slider = row.querySelector('input[type="range"]');
            if (checkbox.checked) {
                loras.push({
                    name: row.dataset.loraName,
                    filename: row.dataset.loraFilename,
                    strength: parseInt(slider.value, 10) / 100,
                });
            }
        }
        try {
            await api._post('/api/story/loras', { loras });
            if (this.storyData) this.storyData.style_loras = loras;
        } catch (error) {
            console.error('Failed to save LoRA selection:', error);
        }
    }

    async _uploadLora(file) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            await fetch('/api/loras/upload', { method: 'POST', body: formData });
            this._renderLoraSection();
        } catch (error) {
            console.error('LoRA upload failed:', error);
        }
    }

    async _searchCivitai() {
        const query = document.getElementById('civitai-search').value.trim();
        if (!query) return;

        const container = document.getElementById('civitai-results');
        container.innerHTML = '<span class="hint">Searching...</span>';

        try {
            const data = await api._get(`/api/civitai/search?query=${encodeURIComponent(query)}`);
            container.innerHTML = '';

            if (!data.results || data.results.length === 0) {
                container.innerHTML = '<span class="hint">No results</span>';
                return;
            }

            for (const result of data.results) {
                const card = document.createElement('div');
                card.className = 'civitai-card';

                if (result.preview_url) {
                    const img = document.createElement('img');
                    img.src = result.preview_url;
                    img.alt = result.name;
                    img.className = 'civitai-preview';
                    card.appendChild(img);
                }

                const info = document.createElement('div');
                info.className = 'civitai-info';

                const name = document.createElement('div');
                name.className = 'civitai-name';
                name.textContent = result.name;
                info.appendChild(name);

                const meta = document.createElement('div');
                meta.className = 'civitai-meta';
                meta.textContent = `${result.size_mb}MB · ${result.downloads} downloads`;
                info.appendChild(meta);

                const dlBtn = document.createElement('button');
                dlBtn.className = 'small-btn';
                dlBtn.textContent = 'Download';
                dlBtn.addEventListener('click', async () => {
                    dlBtn.disabled = true;
                    dlBtn.textContent = 'Downloading...';
                    try {
                        await api._post('/api/civitai/download', {
                            download_url: result.download_url,
                            filename: result.filename,
                        });
                        dlBtn.textContent = 'Downloaded';
                        this._renderLoraSection();
                    } catch (e) {
                        dlBtn.textContent = 'Failed';
                        console.error('Download failed:', e);
                    }
                });
                info.appendChild(dlBtn);

                card.appendChild(info);
                container.appendChild(card);
            }
        } catch (error) {
            container.innerHTML = '<span class="hint">Search failed</span>';
            console.error('Civitai search failed:', error);
        }
    }

    // --- Style References ---

    async _renderStyleReferences() {
        const grid = document.getElementById('style-ref-grid');
        if (!grid) return;
        grid.innerHTML = '';

        const refs = this.storyData?.style_references || [];
        if (refs.length === 0) {
            grid.innerHTML = '<span class="hint">No style references yet</span>';
            return;
        }

        for (const hash of refs) {
            const thumb = document.createElement('div');
            thumb.className = 'style-ref-thumb';

            const img = document.createElement('img');
            img.src = api.contentUrl(hash);
            thumb.appendChild(img);

            const removeBtn = document.createElement('button');
            removeBtn.className = 'style-ref-remove';
            removeBtn.textContent = '\u00d7';
            removeBtn.addEventListener('click', async () => {
                await api._delete(`/api/story/style-references/${hash}`);
                this.storyData.style_references = this.storyData.style_references.filter(h => h !== hash);
                this._renderStyleReferences();
            });
            thumb.appendChild(removeBtn);

            grid.appendChild(thumb);
        }
    }

    async _uploadStyleReference(file) {
        const formData = new FormData();
        formData.append('file', file);
        try {
            const result = await fetch('/api/story/style-references/upload', {
                method: 'POST', body: formData,
            }).then(r => r.json());
            if (this.storyData) {
                if (!this.storyData.style_references) this.storyData.style_references = [];
                if (!this.storyData.style_references.includes(result.content_hash)) {
                    this.storyData.style_references.push(result.content_hash);
                }
            }
            this._renderStyleReferences();
        } catch (error) {
            console.error('Style reference upload failed:', error);
        }
    }

    async _renderAdapterStatus() {
        try {
            const result = await api.getFeedback();
            document.getElementById('adapter-feedback-summary').textContent =
                `${result.positive_count} positive, ${result.negative_count} negative images`;
            document.getElementById('btn-train-adapter-chapter').disabled = !result.can_train;
        } catch (error) {
            document.getElementById('adapter-feedback-summary').textContent = 'No feedback yet';
        }
    }

    async _trainAdapterFromChapterSelect() {
        const btn = document.getElementById('btn-train-adapter-chapter');
        btn.disabled = true;
        btn.textContent = 'Training...';
        try {
            const rank = parseInt(document.getElementById('adapter-rank-chapter').value);
            const epochs = parseInt(document.getElementById('adapter-epochs-chapter').value);
            await api._post('/api/adapter/train', { rank, epochs });
            btn.textContent = 'Trained!';
            setTimeout(() => { btn.textContent = 'Train Adapter'; btn.disabled = false; }, 3000);
        } catch (error) {
            console.error('Training failed:', error);
            btn.textContent = 'Failed';
            setTimeout(() => { btn.textContent = 'Train Adapter'; btn.disabled = false; }, 3000);
        }
    }

    _renderChapterGrid() {
        const grid = document.getElementById('chapter-grid');
        grid.innerHTML = '';

        if (this.chapters.length === 0) {
            grid.innerHTML = '<span class="saved-story-meta">No chapters yet — create one to start</span>';
            return;
        }

        for (let chapterIndex = 0; chapterIndex < this.chapters.length; chapterIndex++) {
            const chapter = this.chapters[chapterIndex];

            // Skip solo chapters — they show on the character screen
            if (chapter.is_solo) continue;

            const card = document.createElement('div');
            card.className = 'chapter-card';

            const title = document.createElement('div');
            title.className = 'chapter-card-title';
            title.textContent = chapter.title || `Chapter ${chapterIndex + 1}`;
            card.appendChild(title);

            const meta = document.createElement('div');
            meta.className = 'chapter-card-meta';
            const pageCount = (chapter.pages || []).length;
            meta.textContent = `${pageCount} page${pageCount !== 1 ? 's' : ''}`;
            card.appendChild(meta);

            // Chapter context fields (editable)
            const fields = document.createElement('div');
            fields.className = 'chapter-card-fields';

            const locationInput = document.createElement('input');
            locationInput.type = 'text';
            locationInput.className = 'chapter-field';
            locationInput.placeholder = 'Default location...';
            locationInput.value = chapter.default_location || '';
            locationInput.addEventListener('click', (e) => e.stopPropagation());
            locationInput.addEventListener('change', async () => {
                await api.updateChapter(chapter.chapter_id, { default_location: locationInput.value });
                chapter.default_location = locationInput.value;
            });
            fields.appendChild(locationInput);

            const timeInput = document.createElement('input');
            timeInput.type = 'text';
            timeInput.className = 'chapter-field';
            timeInput.placeholder = 'Default time of day...';
            timeInput.value = chapter.default_time_of_day || '';
            timeInput.addEventListener('click', (e) => e.stopPropagation());
            timeInput.addEventListener('change', async () => {
                await api.updateChapter(chapter.chapter_id, { default_time_of_day: timeInput.value });
                chapter.default_time_of_day = timeInput.value;
            });
            fields.appendChild(timeInput);

            const synopsisInput = document.createElement('input');
            synopsisInput.type = 'text';
            synopsisInput.className = 'chapter-field';
            synopsisInput.placeholder = 'Synopsis...';
            synopsisInput.value = chapter.synopsis || '';
            synopsisInput.addEventListener('click', (e) => e.stopPropagation());
            synopsisInput.addEventListener('change', async () => {
                await api.updateChapter(chapter.chapter_id, { synopsis: synopsisInput.value });
                chapter.synopsis = synopsisInput.value;
            });
            fields.appendChild(synopsisInput);

            card.appendChild(fields);

            // Show character tags
            const chars = document.createElement('div');
            chars.className = 'chapter-card-characters';
            for (const charId of chapter.character_ids || []) {
                const character = this.characters[charId];
                if (!character) continue;
                const tag = document.createElement('span');
                tag.className = 'chapter-card-char';
                tag.textContent = character.name;
                chars.appendChild(tag);
            }
            card.appendChild(chars);

            card.addEventListener('click', () => this._enterChapter(chapterIndex));
            grid.appendChild(card);
        }
    }

    _enterChapter(chapterIndex) {
        this.currentChapterIndex = chapterIndex;
        this.currentPageIndex = 0;
        this.selectedPanelId = null;

        document.getElementById('chapter-select').classList.add('hidden');
        document.getElementById('app').classList.remove('hidden');
        this._render();
    }

    async _bootstrapStory() {
        try {
            // 1. Default character — a fairy, just us fairies here
            const defaultCharacter = await api.createCharacter({
                name: 'Fairy',
                appearance_prompt: 'small glowing fairy with translucent wings, sparkles, soft light',
                personality_prompt: 'playful, curious, mischievous',
                description: 'A default fairy companion',
                is_temporary: true,
            });

            // 2. Chapter — cascade creates page -> panel -> script
            await api.createChapter({
                title: 'Chapter 1',
                synopsis: '',
                character_ids: [defaultCharacter.character_id],
            });

            // Reload full state — the cascade has created everything
            const storyData = await api.getStory();
            this._loadStoryData(storyData);
        } catch (error) {
            console.error('Failed to bootstrap story:', error);
        }
    }

    _loadStoryData(storyData) {
        this.storyData = storyData;
        this.characters = storyData.characters || {};
        this.chapters = Object.values(storyData.chapters || {});
    }

    // --- Event binding ---

    _bindEvents() {
        // Panel selection
        this.comicRenderer.onPanelSelect = (panelData) => {
            this.selectedPanelId = panelData.panel_id;
            this._clearChat();
            this._loadPanelChat();
            this._renderSidebar();
            this._renderPanelSelection();
        };

        // Page navigation
        document.getElementById('btn-prev-page').addEventListener('click', () => this._navigatePage(-1));
        document.getElementById('btn-next-page').addEventListener('click', () => this._navigatePage(1));
        document.getElementById('btn-add-page').addEventListener('click', () => this._addPage());
        document.getElementById('btn-remove-page').addEventListener('click', () => this._removePage());

        // Panel
        document.getElementById('btn-add-panel').addEventListener('click', () => this._addPanel());

        // Chapter selector
        document.getElementById('btn-chapter-back').addEventListener('click', () => this._showSplash());
        document.getElementById('btn-add-chapter-select').addEventListener('click', () => this._addChapterFromSelect());
        document.getElementById('btn-chapter-manage-chars').addEventListener('click', () => this._showCharacterScreen());
        document.getElementById('btn-train-adapter-chapter').addEventListener('click', () => this._trainAdapterFromChapterSelect());

        // LoRA library
        document.getElementById('lora-upload').addEventListener('change', async (event) => {
            if (event.target.files[0]) await this._uploadLora(event.target.files[0]);
            event.target.value = '';
        });
        document.getElementById('btn-browse-civitai').addEventListener('click', () => {
            const browser = document.getElementById('civitai-browser');
            browser.hidden = !browser.hidden;
        });
        document.getElementById('btn-civitai-search').addEventListener('click', () => this._searchCivitai());
        document.getElementById('civitai-search').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this._searchCivitai();
        });

        // Style references
        document.getElementById('style-ref-upload').addEventListener('change', async (event) => {
            if (event.target.files[0]) await this._uploadStyleReference(event.target.files[0]);
            event.target.value = '';
        });

        // Panel remove
        document.getElementById('btn-remove-panel').addEventListener('click', () => this._removePanel());

        // Character screen
        document.getElementById('btn-char-screen-back').addEventListener('click', () => this._showChapterSelect());
        document.getElementById('btn-char-screen-add').addEventListener('click', () => this._openCharacterManager());
        document.getElementById('new-char-from-image').addEventListener('change', (event) => {
            if (event.target.files[0]) this._newCharacterFromImage(event.target.files[0]);
            event.target.value = '';
        });
        document.getElementById('btn-char-create-confirm').addEventListener('click', () => this._confirmCharacterFromImage());
        document.getElementById('btn-char-create-cancel').addEventListener('click', () => this._cancelCharacterFromImage());
        document.getElementById('btn-generate-solo').addEventListener('click', () => {
            document.getElementById('solo-generate-area').hidden =
                !document.getElementById('solo-generate-area').hidden;
        });
        document.getElementById('btn-solo-go').addEventListener('click', () => this._generateSolo());
        document.getElementById('reference-upload').addEventListener('change', (event) => {
            if (event.target.files.length > 0) this._uploadReferences(event.target.files);
            event.target.value = '';
        });
        document.getElementById('reference-upload-analyze').addEventListener('change', (event) => {
            if (event.target.files[0]) this._uploadAndAnalyze(event.target.files[0]);
            event.target.value = '';
        });
        document.getElementById('btn-apply-analysis').addEventListener('click', () => this._applyAnalysis());
        document.getElementById('btn-close-analysis').addEventListener('click', () => {
            document.getElementById('analysis-result').hidden = true;
        });
        document.getElementById('btn-ref-accept').addEventListener('click', () => this._rateReference(true));
        document.getElementById('btn-ref-reject').addEventListener('click', () => this._rateReference(false));
        document.getElementById('btn-ref-save').addEventListener('click', () => this._saveReferenceLabels());
        document.getElementById('btn-ref-delete').addEventListener('click', () => this._deleteReference());
        document.getElementById('btn-solo-thumbs-up').addEventListener('click', () => this._soloFeedback(true));
        document.getElementById('btn-solo-thumbs-down').addEventListener('click', () => this._soloFeedback(false));
        document.getElementById('btn-solo-regenerate').addEventListener('click', () => this._soloRegenerate());
        document.getElementById('btn-edit-character-detail').addEventListener('click', () => {
            if (this._detailCharacterId) this._startCharacterEdit(this._detailCharacterId);
        });

        // Character manager
        document.getElementById('btn-manage-characters').addEventListener('click', () => this._showCharacterScreen());
        document.getElementById('btn-close-characters').addEventListener('click', () => this._closeCharacterManager());
        document.getElementById('btn-create-character').addEventListener('click', () => this._createCharacter());
        document.getElementById('btn-save-character-edit').addEventListener('click', () => this._saveCharacterEdit());
        document.getElementById('btn-cancel-character-edit').addEventListener('click', () => this._cancelCharacterEdit());

        // Script character dropdown
        document.getElementById('script-character-select').addEventListener('change', (event) => {
            if (event.target.value) this._addScriptToPanel(event.target.value);
        });

        // Panel actions
        document.getElementById('btn-edit-panel').addEventListener('click', () => {
            const panel = this.selectedPanel;
            if (!panel) return;
            // Pass image_url so editor can load the image
            const panelWithUrl = {
                ...panel,
                image_url: panel.image_hash ? api.contentUrl(panel.image_hash) : null,
            };
            this.panelEditor.open(panelWithUrl);
        });

        // Inpainting callback from editor
        this.panelEditor.onApplyEdit = async (panelId, maskBase64, prompt, strength) => {
            const panel = this.currentPage?.panels?.find(p => p.panel_id === panelId);
            if (!panel || !panel.image_hash) return;

            this._showPanelSpinner(panelId, 'Editing...');

            try {
                const negativePrompt = document.getElementById('negative-prompt')?.value || this.defaultNegativePrompt;
                const response = await api.inpaintPanel(panelId, maskBase64, prompt, {
                    negative_prompt: negativePrompt,
                    strength: strength || 0.85,
                });
                panel.image_hash = response.content_hash;
                this._hidePanelSpinner(panelId);
                this._renderPage();
                this._renderPanelSelection();
            } catch (error) {
                console.error('Inpainting failed:', error);
                this._hidePanelSpinner(panelId);
            }
        };

        document.getElementById('btn-generate-panel').addEventListener('click', () => this._generatePanel());
        document.getElementById('btn-save-to-bank').addEventListener('click', () => this._savePanelToBank());

        // Chat
        document.getElementById('btn-send').addEventListener('click', () => this._sendChat());
        document.getElementById('chat-input').addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this._sendChat();
            }
        });

        // Page context
        for (const field of ['page-setting', 'page-mood', 'page-action', 'page-time-of-day', 'page-weather', 'page-lighting']) {
            document.getElementById(field).addEventListener('change', () => this._savePageContext());
        }

        // Panel properties
        document.getElementById('panel-shot-type').addEventListener('change', () => this._savePanelProps());
        document.getElementById('panel-narration').addEventListener('change', () => this._savePanelProps());

        // Feedback
        document.getElementById('btn-thumbs-up').addEventListener('click', () => this._submitFeedback(true));
        document.getElementById('btn-thumbs-down').addEventListener('click', () => this._submitFeedback(false));
        document.getElementById('btn-train-adapter').addEventListener('click', () => this._trainAdapter());
        document.getElementById('btn-review').addEventListener('click', () => this._reviewPanel());
        document.getElementById('btn-apply-review').addEventListener('click', () => this._applyReviewSuggestions());

        // Chat actions
        document.getElementById('btn-suggest-scripts').addEventListener('click', () => this._suggestScripts());
        document.getElementById('btn-react').addEventListener('click', () => this._characterReact());

        // Regenerate
        document.getElementById('btn-regenerate-all').addEventListener('click', () => this._regenerateAll());

        // Header
        document.getElementById('btn-save').addEventListener('click', () => this._saveStory());
        document.getElementById('btn-download').addEventListener('click', () => this._downloadStory());
        document.getElementById('btn-back-to-chapters').addEventListener('click', () => this._showChapterSelect());

        // Splash screen
        document.getElementById('btn-splash-new').addEventListener('click', () => this._newStoryAndEnter());
        document.getElementById('splash-load-file').addEventListener('change', (event) => {
            if (event.target.files[0]) this._loadStoryFile(event.target.files[0]);
            event.target.value = '';
        });

        // Character import
        document.getElementById('import-character-file').addEventListener('change', async (event) => {
            if (event.target.files[0]) await this._importCharacterFile(event.target.files[0]);
            event.target.value = '';
        });
    }

    // --- Rendering ---

    _render() {
        this._renderPage();
        this._renderSidebar();
        this._renderNav();
        this._loadPageContext();
    }

    _loadStorySettings() {
        document.getElementById('story-art-style').value = this.storyData?.art_style || '';
        document.getElementById('story-genre').value = this.storyData?.genre || '';
        document.getElementById('story-synopsis').value = this.storyData?.synopsis || '';

        // Wire change events (only once)
        if (!this._storySettingsWired) {
            for (const field of ['story-art-style', 'story-genre', 'story-synopsis']) {
                document.getElementById(field).addEventListener('change', () => this._saveStorySettings());
            }
            this._storySettingsWired = true;
        }
    }

    async _saveStorySettings() {
        try {
            const updated = await api.put('/api/story', {
                art_style: document.getElementById('story-art-style').value,
                genre: document.getElementById('story-genre').value,
                synopsis: document.getElementById('story-synopsis').value,
            });
            if (this.storyData) {
                this.storyData.art_style = updated.art_style;
                this.storyData.genre = updated.genre;
                this.storyData.synopsis = updated.synopsis;
            }
        } catch (error) {
            console.error('Failed to save story settings:', error);
        }
    }

    _loadPageContext() {
        const page = this.currentPage;
        document.getElementById('page-setting').value = page?.setting || '';
        document.getElementById('page-mood').value = page?.mood || '';
        document.getElementById('page-action').value = page?.action_context || '';
        document.getElementById('page-time-of-day').value = page?.time_of_day || '';
        document.getElementById('page-weather').value = page?.weather || '';
        document.getElementById('page-lighting').value = page?.lighting || '';
    }

    async _savePageContext() {
        const page = this.currentPage;
        if (!page) return;
        try {
            await api.updatePage(page.page_id, {
                setting: document.getElementById('page-setting').value,
                mood: document.getElementById('page-mood').value,
                action_context: document.getElementById('page-action').value,
                time_of_day: document.getElementById('page-time-of-day').value,
                weather: document.getElementById('page-weather').value,
                lighting: document.getElementById('page-lighting').value,
            });
            page.setting = document.getElementById('page-setting').value;
            page.mood = document.getElementById('page-mood').value;
            page.action_context = document.getElementById('page-action').value;
            page.time_of_day = document.getElementById('page-time-of-day').value;
            page.weather = document.getElementById('page-weather').value;
            page.lighting = document.getElementById('page-lighting').value;
        } catch (error) {
            console.error('Failed to save page context:', error);
        }
    }

    _loadPanelProps() {
        const panel = this.selectedPanel;
        document.getElementById('panel-shot-type').value = panel?.shot_type || '';
        document.getElementById('panel-narration').value = panel?.narration || '';
    }

    async _savePanelProps() {
        const panel = this.selectedPanel;
        if (!panel) return;
        try {
            const shotType = document.getElementById('panel-shot-type').value;
            const narration = document.getElementById('panel-narration').value;
            await api.put(`/api/panels/${panel.panel_id}`, {
                shot_type: shotType,
                narration: narration,
            });
            panel.shot_type = shotType;
            panel.narration = narration;
        } catch (error) {
            console.error('Failed to save panel properties:', error);
        }
    }

    _renderPage() {
        const page = this.currentPage;
        if (!page || !page.panels || page.panels.length === 0) {
            this.comicRenderer.renderPlaceholderPage(0);
            return;
        }

        // Build page data for renderer with image URLs
        const pageData = {
            page_id: page.page_id,
            template: this._layoutTemplate(page.panels.length),
            panels: page.panels.map(panel => ({
                panel_id: panel.panel_id,
                image_url: panel.image_hash ? api.contentUrl(panel.image_hash) : null,
                image_path: null,
                video_path: null,
                is_animated: panel.is_animated || false,
                dialogue: this._panelDialogue(panel),
                narration: panel.narration || '',
            })),
        };
        this.comicRenderer.renderPage(pageData);
        this._renderPanelSelection();
    }

    _panelDialogue(panel) {
        if (!panel.scripts) return [];
        return Object.values(panel.scripts)
            .filter(script => script.dialogue)
            .map(script => ({
                character: (this.characters[script.character_id] || {}).name || script.character_id,
                text: script.dialogue,
            }));
    }

    _panelGenerationSize(panelId) {
        /**
         * Measure the panel element's aspect ratio and return
         * generation dimensions that match. Rounds to multiples of 8
         * (SDXL requirement). Base dimension is 512.
         */
        const element = document.querySelector(`.comic-panel[data-panel-id="${panelId}"]`);
        if (!element) return { width: 768, height: 512 };

        const rect = element.getBoundingClientRect();
        const aspect = rect.width / rect.height;

        // Base dimension 512, scale the other axis to match aspect ratio
        let width, height;
        if (aspect >= 1) {
            // Wider than tall
            height = 512;
            width = Math.round(512 * aspect);
        } else {
            // Taller than wide
            width = 512;
            height = Math.round(512 / aspect);
        }

        // Round to nearest multiple of 8
        width = Math.round(width / 8) * 8;
        height = Math.round(height / 8) * 8;

        // Clamp to reasonable range
        width = Math.max(512, Math.min(1024, width));
        height = Math.max(512, Math.min(1024, height));

        return { width, height };
    }

    _layoutTemplate(panelCount) {
        if (panelCount <= 1) return 'single';
        if (panelCount === 2) return 'two_equal';
        if (panelCount === 3) return 'two_plus_tall';   // 2 left + 1 tall right
        if (panelCount === 4) return 'grid_2x2';
        if (panelCount === 5) return 'grid_2x2_plus_tall'; // 2x2 left + 1 tall right
        if (panelCount === 6) return 'grid_3x2';
        if (panelCount === 7) return 'grid_3x2_plus_tall'; // 3x2 left + 1 tall right
        return 'grid_4x2';
    }

    _renderPanelSelection() {
        if (this.selectedPanelId) {
            this.comicRenderer._selectPanel(this.selectedPanelId);
        }
    }

    _renderSidebar() {
        this._renderChapterSection();

        const panelSection = document.getElementById('panel-section');
        const chatSection = document.getElementById('chat-section');
        const noPanel = document.getElementById('no-panel-message');

        if (this.selectedPanel) {
            panelSection.hidden = false;
            chatSection.hidden = false;
            noPanel.hidden = true;
            this._renderPanelScripts();
            this._renderScriptCharacterDropdown();
            this._renderChatCharacterDropdown();
            this._loadPanelProps();
            document.getElementById('btn-generate-panel').disabled =
                this.isGenerating || !this._panelHasScripts();
            document.getElementById('btn-save-to-bank').disabled =
                !this.selectedPanel?.image_hash;

            // Show feedback if panel has an image
            const feedbackEl = document.getElementById('panel-feedback');
            if (this.selectedPanel?.image_hash) {
                feedbackEl.removeAttribute('hidden');
            } else {
                feedbackEl.setAttribute('hidden', '');
            }
            document.getElementById('btn-thumbs-up').classList.remove('selected-up');
            document.getElementById('btn-thumbs-down').classList.remove('selected-down');
            document.getElementById('review-result').hidden = true;
            // Check if this panel's image already has feedback
            if (this.selectedPanel?.image_hash && this.selectedPanel?._feedback !== undefined) {
                document.getElementById('btn-thumbs-up').classList.toggle('selected-up', this.selectedPanel._feedback === true);
                document.getElementById('btn-thumbs-down').classList.toggle('selected-down', this.selectedPanel._feedback === false);
            }
            this._updateFeedbackCount();

            // Load negative prompt
            const negPromptEl = document.getElementById('negative-prompt');
            if (negPromptEl && !negPromptEl.value) {
                negPromptEl.value = this.defaultNegativePrompt;
            }
        } else {
            panelSection.hidden = true;
            chatSection.hidden = true;
            noPanel.hidden = false;
        }
    }

    _renderChapterSection() {
        const chapter = this.currentChapter;
        document.getElementById('chapter-title').textContent = chapter ? chapter.title : 'No Chapter';

        // Show chapter characters
        const container = document.getElementById('chapter-characters');
        container.innerHTML = '';

        if (!chapter) return;

        for (const characterId of chapter.character_ids || []) {
            const character = this.characters[characterId];
            if (!character) continue;

            const element = document.createElement('div');
            element.className = 'character-tag';
            element.textContent = character.name;
            element.title = 'Click to view character bank';
            element.style.cursor = 'pointer';
            element.addEventListener('click', () => this._openCharacterDetail(characterId));
            container.appendChild(element);
        }
    }

    _renderPanelScripts() {
        const panel = this.selectedPanel;
        const container = document.getElementById('panel-scripts');
        container.innerHTML = '';

        if (!panel || !panel.scripts) return;

        for (const [characterId, script] of Object.entries(panel.scripts)) {
            const character = this.characters[characterId];
            const scriptElement = this._createScriptEditor(character, script);
            container.appendChild(scriptElement);
        }
    }

    _createScriptEditor(character, script) {
        const wrapper = document.createElement('div');
        wrapper.className = 'script-entry';

        const header = document.createElement('div');
        header.className = 'script-header';

        const headerName = document.createElement('span');
        headerName.textContent = character ? character.name : script.character_id;
        header.appendChild(headerName);

        const panel = this.selectedPanel;
        const scriptCount = panel ? Object.keys(panel.scripts || {}).length : 0;

        const removeBtn = document.createElement('button');
        removeBtn.className = 'script-remove-btn';
        removeBtn.textContent = '\u00d7';
        removeBtn.title = scriptCount <= 1 ? 'Cannot remove last character' : 'Remove from panel';
        removeBtn.disabled = scriptCount <= 1;
        removeBtn.addEventListener('click', () => this._removeScriptFromPanel(script.character_id));
        header.appendChild(removeBtn);

        wrapper.appendChild(header);

        const fields = [
            { key: 'dialogue', placeholder: 'Dialogue...', value: script.dialogue },
            { key: 'action', placeholder: 'Action...', value: script.action },
            { key: 'emotion', placeholder: 'Emotion...', value: script.emotion },
            { key: 'pose', placeholder: 'Pose (e.g. standing, sitting, crouching)...', value: script.pose },
            { key: 'outfit', placeholder: 'Outfit (overrides default)...', value: script.outfit },
            { key: 'direction', placeholder: 'Direction...', value: script.direction },
            { key: 'negative_prompt', placeholder: 'Negative (e.g. animal ears, tail)...', value: script.negative_prompt },
        ];

        for (const field of fields) {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'script-field';
            input.placeholder = field.placeholder;
            input.value = field.value || '';
            input.dataset.scriptId = script.script_id;
            input.dataset.field = field.key;
            input.addEventListener('change', (event) => this._updateScript(event));
            wrapper.appendChild(input);
        }

        return wrapper;
    }

    _renderScriptCharacterDropdown() {
        const select = document.getElementById('script-character-select');
        select.innerHTML = '<option value="">Add chapter character to panel...</option>';

        const chapter = this.currentChapter;
        const panel = this.selectedPanel;
        if (!chapter || !panel) return;

        // Only chapter characters can be added — they enter through the chapter, not the panel
        const existingIds = panel.scripts ? Object.keys(panel.scripts) : [];
        let availableCount = 0;

        for (const characterId of chapter.character_ids || []) {
            if (existingIds.includes(characterId)) continue;
            const character = this.characters[characterId];
            if (!character) continue;

            const option = document.createElement('option');
            option.value = characterId;
            option.textContent = character.name;
            select.appendChild(option);
            availableCount++;
        }

        if (availableCount === 0) {
            select.innerHTML = '<option value="">All chapter characters added</option>';
        }
    }

    _panelHasScripts() {
        const panel = this.selectedPanel;
        return panel && panel.scripts && Object.keys(panel.scripts).length > 0;
    }

    _renderNav() {
        const pages = this.currentPages;
        const pageNumber = this.currentPageIndex + 1;
        const totalPages = pages.length;
        const panelCount = this.currentPage ? (this.currentPage.panels || []).length : 0;

        document.getElementById('page-indicator').textContent = `Page ${pageNumber} / ${totalPages}`;
        document.getElementById('btn-prev-page').disabled = this.currentPageIndex <= 0;
        document.getElementById('btn-next-page').disabled = this.currentPageIndex >= pages.length - 1;
        document.getElementById('btn-remove-page').disabled = pages.length <= 1;
        document.getElementById('panel-count-indicator').textContent = `${panelCount} panel${panelCount !== 1 ? 's' : ''}`;
        document.getElementById('btn-add-panel').disabled = !this.currentPage || panelCount >= 8;
        document.getElementById('btn-remove-panel').disabled = !this.selectedPanelId || panelCount <= 1;
        // Chapter title in sidebar
        const chapterTitle = document.getElementById('chapter-title');
        if (chapterTitle && this.currentChapter) {
            chapterTitle.textContent = this.currentChapter.title || 'Chapter';
        }
    }

    // --- Actions ---

    _navigatePage(direction) {
        const newIndex = this.currentPageIndex + direction;
        if (newIndex < 0 || newIndex >= this.currentPages.length) return;
        this.currentPageIndex = newIndex;
        this.selectedPanelId = null;
        this._clearChat();
        this._render();
    }

    async _addPage() {
        const chapter = this.currentChapter;
        if (!chapter) return;

        try {
            const page = await api.createPage(chapter.chapter_id);
            chapter.pages.push(page);
            this.currentPageIndex = chapter.pages.length - 1;
            this.selectedPanelId = null;
            this._render();
        } catch (error) {
            console.error('Failed to add page:', error);
        }
    }

    _removePage() {
        // TODO: implement via API
    }

    async _addPanel() {
        const page = this.currentPage;
        if (!page) return;

        try {
            const panel = await api.createPanel(page.page_id);
            if (!page.panels) page.panels = [];
            page.panels.push(panel);
            this.selectedPanelId = panel.panel_id;
            this._renderPage();
            this._renderSidebar();
            this._renderNav();
        } catch (error) {
            console.error('Failed to add panel:', error);
        }
    }

    _showPanelSpinner(panelId, text = 'Generating...') {
        const panelElement = document.querySelector(`.comic-panel[data-panel-id="${panelId}"]`);
        if (!panelElement) return;
        const existing = panelElement.querySelector('.panel-spinner');
        if (existing) existing.remove();

        const spinner = document.createElement('div');
        spinner.className = 'panel-spinner';

        const circle = document.createElement('div');
        circle.className = 'panel-spinner-circle';
        spinner.appendChild(circle);

        const label = document.createElement('span');
        label.className = 'panel-spinner-text';
        label.textContent = text;
        spinner.appendChild(label);

        panelElement.appendChild(spinner);
    }

    _hidePanelSpinner(panelId) {
        const panelElement = document.querySelector(`.comic-panel[data-panel-id="${panelId}"]`);
        if (!panelElement) return;
        const spinner = panelElement.querySelector('.panel-spinner');
        if (spinner) spinner.remove();
    }

    _removePanel() {
        const page = this.currentPage;
        if (!page || !this.selectedPanelId) return;
        if ((page.panels || []).length <= 1) return;

        page.panels = page.panels.filter(p => p.panel_id !== this.selectedPanelId);
        this.selectedPanelId = null;
        this._render();
    }

    async _addChapterFromSelect() {
        const characterIds = Object.keys(this.characters);
        if (characterIds.length === 0) {
            alert('Create at least one character first');
            return;
        }

        try {
            const chapter = await api.createChapter({
                title: `Chapter ${this.chapters.length + 1}`,
                synopsis: '',
                character_ids: characterIds,
            });
            this.chapters.push(chapter);
            this._renderChapterGrid();
        } catch (error) {
            console.error('Failed to add chapter:', error);
        }
    }

    async _addScriptToPanel(characterId) {
        const panel = this.selectedPanel;
        if (!panel) return;

        try {
            const script = await api.createScript({
                panel_id: panel.panel_id,
                character_id: characterId,
            });

            if (!panel.scripts) panel.scripts = {};
            panel.scripts[characterId] = script;

            this._renderSidebar();
        } catch (error) {
            console.error('Failed to add script:', error);
            alert(error.message);
        }

        // Reset dropdown
        document.getElementById('script-character-select').value = '';
    }

    async _removeScriptFromPanel(characterId) {
        const panel = this.selectedPanel;
        if (!panel || !panel.scripts) return;

        // Cannot remove the last script — would break the hierarchy
        if (Object.keys(panel.scripts).length <= 1) return;

        // Remove on backend
        const script = panel.scripts[characterId];
        if (script && script.script_id) {
            try {
                await api._delete(`/api/scripts/${script.script_id}`);
            } catch (error) {
                console.error('Failed to remove script:', error);
                return;
            }
        }

        delete panel.scripts[characterId];
        this._renderSidebar();
        this._renderPage();
        this._renderPanelSelection();
    }

    async _updateScript(event) {
        const scriptId = event.target.dataset.scriptId;
        const field = event.target.dataset.field;
        const value = event.target.value;

        try {
            const updated = await api.updateScript(scriptId, { [field]: value });

            // Update local state
            const panel = this.selectedPanel;
            if (panel && panel.scripts) {
                for (const script of Object.values(panel.scripts)) {
                    if (script.script_id === scriptId) {
                        Object.assign(script, updated);
                        break;
                    }
                }
            }

            // Enable generate if panel now has scripts
            document.getElementById('btn-generate-panel').disabled =
                this.isGenerating || !this._panelHasScripts();
            document.getElementById('btn-save-to-bank').disabled =
                !this.selectedPanel?.image_hash;
        } catch (error) {
            console.error('Failed to update script:', error);
        }
    }

    async _generatePanel() {
        const panel = this.selectedPanel;
        if (!panel || this.isGenerating) return;

        const panelId = panel.panel_id;
        this.isGenerating = true;
        const generateButton = document.getElementById('btn-generate-panel');
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';

        this._showPanelSpinner(panelId, 'Generating...');

        try {
            const negativePrompt = document.getElementById('negative-prompt')?.value || this.defaultNegativePrompt;
            const panelSize = this._panelGenerationSize(panelId);
            const response = await api.generatePanel(panelId, {
                negative_prompt: negativePrompt,
                width: panelSize.width,
                height: panelSize.height,
            });

            console.log('=== Generation Complete ===');
            console.log('Method:', response.prompt_method);
            console.log('Direct prompt:', response.prompt_direct);
            if (response.prompt_llm_input) console.log('LLM input:', response.prompt_llm_input);
            if (response.prompt_llm_output) console.log('LLM output:', response.prompt_llm_output);
            console.log('Final prompt:', response.prompt_used);
            console.log('Negative:', response.negative_prompt_used);
            console.log('Characters:', response.characters_used);

            panel.image_hash = response.content_hash;
            panel._last_prompt = response.prompt_used;
            panel._last_negative = response.negative_prompt_used;
            this._hidePanelSpinner(panelId);
            this._renderPage();
            this._renderPanelSelection();
        } catch (error) {
            console.error('Generation failed:', error);
            this._hidePanelSpinner(panelId);
        } finally {
            this.isGenerating = false;
            generateButton.disabled = false;
            generateButton.textContent = 'Generate';
        }
    }

    // --- Script chat ---

    _clearChat() {
        document.getElementById('chat-history').innerHTML = '';
        this._chatHistory = [];
    }

    _loadPanelChat() {
        const panel = this.selectedPanel;
        if (!panel) return;

        // Show narration
        if (panel.narration) {
            this._addChatMessage('Narrator', panel.narration, 'narrator');
        }

        // Show each character's dialogue
        if (panel.scripts) {
            for (const [characterId, script] of Object.entries(panel.scripts)) {
                const character = this.characters[characterId];
                const name = character ? character.name : characterId;
                if (script.dialogue) {
                    this._addChatMessage(name, script.dialogue, 'character');
                }
            }
        }
    }

    _renderChatCharacterDropdown() {
        const select = document.getElementById('chat-character-select');
        select.innerHTML = '<option value="narrator">Narrator</option>';

        const panel = this.selectedPanel;
        if (!panel || !panel.scripts) return;

        // Show characters that have scripts in this panel
        for (const characterId of Object.keys(panel.scripts)) {
            const character = this.characters[characterId];
            if (!character) continue;
            const option = document.createElement('option');
            option.value = characterId;
            option.textContent = character.name;
            select.appendChild(option);
        }
    }

    async _sendChat() {
        const input = document.getElementById('chat-input');
        const text = input.value.trim();
        if (!text || !this.selectedPanel) return;

        const characterSelect = document.getElementById('chat-character-select');
        const characterId = characterSelect.value;
        const characterName = characterSelect.options[characterSelect.selectedIndex].textContent;
        const isNarrator = characterId === 'narrator';
        const panel = this.selectedPanel;

        input.value = '';

        if (isNarrator) {
            this._addChatMessage('Narrator', text, 'narrator');
            panel.narration = (panel.narration ? panel.narration + ' ' : '') + text;
            this._renderPage();
            this._renderPanelSelection();
        } else {
            // User message
            this._addChatMessage('You', text, 'user');

            // Get character response from LLM
            try {
                const response = await api.chatWithCharacter(
                    characterId, text, panel.panel_id, this._chatHistory,
                );

                this._addChatMessage(response.character_name, response.response, 'character');

                // Store in chat history for context
                if (!this._chatHistory) this._chatHistory = [];
                this._chatHistory.push({ role: 'user', content: text });
                this._chatHistory.push({ role: 'assistant', content: response.response });
            } catch (error) {
                this._addChatMessage('System', 'LLM not available — type dialogue directly', 'narrator');
                // Fallback: treat as direct dialogue
                if (panel.scripts && panel.scripts[characterId]) {
                    const script = panel.scripts[characterId];
                    const newDialogue = script.dialogue ? `${script.dialogue} ${text}` : text;
                    try {
                        const updated = await api.updateScript(script.script_id, { dialogue: newDialogue });
                        Object.assign(script, updated);
                        this._renderPanelScripts();
                    } catch (err) {
                        console.error('Failed to update script:', err);
                    }
                }
            }

            this._renderPage();
            this._renderPanelSelection();
        }
    }

    async _suggestScripts() {
        const panel = this.selectedPanel;
        const characterSelect = document.getElementById('chat-character-select');
        const characterId = characterSelect.value;
        if (!panel || characterId === 'narrator') return;

        this._addChatMessage('System', 'Asking for suggestions...', 'narrator');

        try {
            const result = await api.suggestScripts(characterId, panel.panel_id);
            const s = result.suggestions;

            const response = [
                s.dialogue ? `"${s.dialogue}"` : null,
                s.action ? `*${s.action}*` : null,
                s.emotion ? `Emotion: ${s.emotion}` : null,
                s.direction ? `Direction: ${s.direction}` : null,
            ].filter(Boolean).join('\n');

            this._addChatMessage(this.characters[characterId]?.name || characterId, response, 'character', {
                suggestion: s,
                characterId: characterId,
            });
        } catch (error) {
            this._addChatMessage('System', 'Could not get suggestions — is Ollama running?', 'narrator');
        }
    }

    async _characterReact() {
        const panel = this.selectedPanel;
        const characterSelect = document.getElementById('chat-character-select');
        const characterId = characterSelect.value;
        if (!panel || characterId === 'narrator') return;

        this._addChatMessage('System', 'Getting reaction...', 'narrator');

        try {
            const result = await api.characterReact(characterId, panel.panel_id);
            this._addChatMessage(this.characters[characterId]?.name || characterId, result.reaction, 'character');
        } catch (error) {
            this._addChatMessage('System', 'Could not get reaction — is Ollama running?', 'narrator');
        }
    }

    async _useAsDialogue(characterId, text) {
        const panel = this.selectedPanel;
        if (!panel || !panel.scripts || !panel.scripts[characterId]) return;

        const script = panel.scripts[characterId];
        try {
            const updated = await api.updateScript(script.script_id, { dialogue: text });
            Object.assign(script, updated);
            this._renderPanelScripts();
            this._renderPage();
            this._renderPanelSelection();
        } catch (error) {
            console.error('Failed to set dialogue:', error);
        }
    }

    async _applyScriptSuggestion(characterId, suggestion) {
        const panel = this.selectedPanel;
        if (!panel || !panel.scripts || !panel.scripts[characterId]) return;

        const script = panel.scripts[characterId];
        try {
            const updated = await api.updateScript(script.script_id, {
                dialogue: suggestion.dialogue || script.dialogue,
                action: suggestion.action || script.action,
                emotion: suggestion.emotion || script.emotion,
                direction: suggestion.direction || script.direction,
            });
            Object.assign(script, updated);
            this._renderPanelScripts();
            this._renderPage();
            this._renderPanelSelection();
            this._addChatMessage('System', 'Applied suggestion to script', 'narrator');
        } catch (error) {
            console.error('Failed to apply suggestion:', error);
        }
    }

    _addChatMessage(sender, text, type, metadata = null) {
        const history = document.getElementById('chat-history');
        const message = document.createElement('div');
        message.className = `chat-message ${type}`;

        const senderLabel = document.createElement('div');
        senderLabel.className = 'sender';
        senderLabel.textContent = sender;
        message.appendChild(senderLabel);

        const content = document.createElement('div');
        content.className = 'chat-message-text';
        content.textContent = text;
        message.appendChild(content);

        // Action buttons for all non-system messages
        if (type !== 'narrator' || sender !== 'System') {
            const actions = document.createElement('div');
            actions.className = 'chat-message-actions';

            const characterSelect = document.getElementById('chat-character-select');
            const charId = characterSelect?.value;

            // Edit button — makes text editable
            const editBtn = document.createElement('button');
            editBtn.textContent = 'Edit';
            editBtn.addEventListener('click', () => {
                if (content.contentEditable === 'true') {
                    content.contentEditable = 'false';
                    content.classList.remove('editing');
                    editBtn.textContent = 'Edit';
                } else {
                    content.contentEditable = 'true';
                    content.classList.add('editing');
                    content.focus();
                    editBtn.textContent = 'Done';
                }
            });
            actions.appendChild(editBtn);

            if (type === 'character' && this.selectedPanel) {
                // Use as dialogue
                const useBtn = document.createElement('button');
                useBtn.textContent = 'Dialogue';
                useBtn.addEventListener('click', () => {
                    const currentText = content.textContent
                        .replace(/\*[^*]*\*/g, '').replace(/"/g, '').trim();
                    this._useAsDialogue(charId, currentText);
                });
                actions.appendChild(useBtn);

                // Apply all if structured suggestion
                if (metadata?.suggestion && metadata?.characterId) {
                    const applyBtn = document.createElement('button');
                    applyBtn.textContent = 'Apply';
                    applyBtn.addEventListener('click', () =>
                        this._applyScriptSuggestion(
                            metadata.characterId, metadata.suggestion
                        ));
                    actions.appendChild(applyBtn);
                }
            }

            // Delete
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Del';
            deleteBtn.addEventListener('click', () => message.remove());
            actions.appendChild(deleteBtn);

            message.appendChild(actions);
        }

        history.appendChild(message);
        history.scrollTop = history.scrollHeight;
    }

    async _submitFeedback(accepted) {
        const panel = this.selectedPanel;
        if (!panel || !panel.image_hash) return;

        const characterIds = Object.keys(panel.scripts || {});
        try {
            const result = await api.submitFeedback(
                panel.image_hash,
                panel._last_prompt || '',
                accepted,
                characterIds,
                panel.panel_id,
            );

            // Store on panel so it persists when switching
            panel._feedback = accepted;

            // Flash the button briefly to confirm, don't toggle
            const btn = accepted
                ? document.getElementById('btn-thumbs-up')
                : document.getElementById('btn-thumbs-down');
            btn.classList.add(accepted ? 'selected-up' : 'selected-down');
            setTimeout(() => btn.classList.remove(accepted ? 'selected-up' : 'selected-down'), 500);

            this._updateFeedbackCount();
            document.getElementById('btn-train-adapter').disabled = !result.can_train;

            console.log(`Feedback: ${accepted ? 'POSITIVE' : 'NEGATIVE'} — ${result.positive_count} up, ${result.negative_count} down`);
        } catch (error) {
            console.error('Failed to submit feedback:', error);
        }
    }

    async _updateFeedbackCount() {
        try {
            const result = await api.getFeedback();
            const countEl = document.getElementById('feedback-count');
            countEl.textContent = `${result.positive_count}↑ ${result.negative_count}↓`;
            document.getElementById('btn-train-adapter').disabled = !result.can_train;
        } catch (error) {
            // Silently fail — not critical
        }
    }

    async _trainAdapter() {
        const btn = document.getElementById('btn-train-adapter');
        btn.disabled = true;
        btn.textContent = 'Training...';

        try {
            const rank = parseInt(document.getElementById('adapter-rank').value);
            const epochs = parseInt(document.getElementById('adapter-epochs').value);
            const result = await api._post('/api/adapter/train', { rank, epochs });
            btn.textContent = 'Trained!';
            console.log('Adapter trained:', result.adapter_hash);
            setTimeout(() => { btn.textContent = 'Train'; btn.disabled = false; }, 3000);
        } catch (error) {
            console.error('Training failed:', error);
            btn.textContent = 'Failed';
            setTimeout(() => { btn.textContent = 'Train'; btn.disabled = false; }, 3000);
        }
    }

    async _reviewPanel() {
        const panel = this.selectedPanel;
        if (!panel || !panel.image_hash) return;

        const reviewBtn = document.getElementById('btn-review');
        reviewBtn.disabled = true;
        reviewBtn.textContent = 'Reviewing...';

        // Clear pending suggestions
        this._pendingReviewSuggestions = {};

        try {
            const result = await api._post(`/api/review/${panel.panel_id}`, {});

            const reviewEl = document.getElementById('review-result');
            reviewEl.hidden = false;

            const scorePercent = Math.round(result.match_score * 100);
            const scoreColor = scorePercent > 70 ? '#4c4' : scorePercent > 40 ? '#cc4' : '#c44';

            reviewEl.querySelector('.review-score').innerHTML =
                `Match: <span style="color:${scoreColor}">${scorePercent}%</span>`;
            reviewEl.querySelector('.review-caption').textContent =
                `AI sees: ${result.reverse_caption}`;
            reviewEl.querySelector('.review-differences').textContent =
                result.differences.length > 0
                    ? `Mismatches: ${result.differences.join(', ')}`
                    : '';
            reviewEl.querySelector('.review-suggestion').textContent =
                result.suggestion ? `Suggestion: ${result.suggestion}` : '';

            console.log('=== Review Result ===');
            console.log('Original prompt:', result.original_prompt);
            console.log('AI caption:', result.reverse_caption);
            console.log('Score:', result.match_score);
            console.log('Differences:', result.differences);
            console.log('Suggestion:', result.suggestion);

            // Ask LLM to extract structured suggestions per character
            if (result.reverse_caption && panel.scripts) {
                try {
                    const characterIds = Object.keys(panel.scripts);
                    for (const charId of characterIds) {
                        const suggestResult = await api.suggestScripts(
                            charId, panel.panel_id
                        );
                        if (suggestResult.suggestions) {
                            const s = suggestResult.suggestions;
                            this._pendingReviewSuggestions[charId] = s;

                            const charName = this.characters[charId]?.name || charId;
                            const parts = [
                                s.action ? `*${s.action}*` : null,
                                s.dialogue ? `"${s.dialogue}"` : null,
                                s.emotion ? `Emotion: ${s.emotion}` : null,
                                s.direction ? `Direction: ${s.direction}` : null,
                                s.pose ? `Pose: ${s.pose}` : null,
                            ].filter(Boolean);
                            if (parts.length > 0) {
                                this._addChatMessage(
                                    `Review: ${charName}`,
                                    parts.join('\n'),
                                    'character',
                                    { suggestion: s, characterId: charId }
                                );
                            }
                        }
                    }
                } catch (error) {
                    console.warn('Auto-suggest from review failed:', error);
                }
            }

            // Show Apply button if we have suggestions
            const applyBtn = document.getElementById('btn-apply-review');
            applyBtn.hidden = Object.keys(this._pendingReviewSuggestions).length === 0;

        } catch (error) {
            console.error('Review failed:', error);
        } finally {
            reviewBtn.disabled = false;
            reviewBtn.textContent = 'Review';
        }
    }

    async _applyReviewSuggestions() {
        const panel = this.selectedPanel;
        if (!panel || !this._pendingReviewSuggestions) return;

        const applyBtn = document.getElementById('btn-apply-review');
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';

        try {
            // Apply suggestions to each character's script
            for (const [charId, suggestion] of Object.entries(this._pendingReviewSuggestions)) {
                const script = panel.scripts[charId];
                if (!script) continue;

                const updates = {};
                if (suggestion.action) updates.action = suggestion.action;
                if (suggestion.emotion) updates.emotion = suggestion.emotion;
                if (suggestion.direction) updates.direction = suggestion.direction;
                if (suggestion.dialogue) updates.dialogue = suggestion.dialogue;
                if (suggestion.pose) updates.pose = suggestion.pose;

                if (Object.keys(updates).length > 0) {
                    await api.updateScript(script.script_id, updates);
                    Object.assign(script, updates);
                }
            }

            // Re-render the scripts to show updated values
            this._renderPanelScripts();
            this._pendingReviewSuggestions = {};
            applyBtn.hidden = true;

        } catch (error) {
            console.error('Failed to apply suggestions:', error);
        } finally {
            applyBtn.disabled = false;
            applyBtn.textContent = 'Apply Suggestions';
        }
    }

    async _regenerateAll() {
        const keepGood = confirm(
            'Keep images you rated as good?\n\n'
            + 'OK = Keep good images, only regenerate unrated/bad ones\n'
            + 'Cancel = Regenerate everything'
        );

        const btn = document.getElementById('btn-regenerate-all');
        btn.disabled = true;
        btn.textContent = 'Checking...';

        try {
            const plan = await api._post('/api/regenerate-all', {
                keep_good: keepGood,
            });

            const toRegen = plan.to_regenerate;
            const kept = plan.kept;

            if (toRegen.length === 0) {
                btn.textContent = 'All good!';
                setTimeout(() => {
                    btn.textContent = 'Regenerate';
                    btn.disabled = false;
                }, 2000);
                return;
            }

            const proceed = confirm(
                `Regenerate ${toRegen.length} panels?\n`
                + `${kept.length} panels kept (rated good).\n\n`
                + `This will take about ${toRegen.length * 30} seconds.`
            );

            if (!proceed) {
                btn.textContent = 'Regenerate';
                btn.disabled = false;
                return;
            }

            btn.textContent = `0/${toRegen.length}`;

            for (let i = 0; i < toRegen.length; i++) {
                const panelId = toRegen[i].panel_id;
                btn.textContent = `${i + 1}/${toRegen.length}`;
                this._showPanelSpinner(panelId, `Regenerating ${i + 1}/${toRegen.length}...`);

                try {
                    const negativePrompt = document.getElementById('negative-prompt')?.value || this.defaultNegativePrompt;
                    const response = await api.generatePanel(panelId, {
                        negative_prompt: negativePrompt,
                    });

                    // Update local panel data
                    for (const ch of this.chapters) {
                        for (const pg of ch.pages || []) {
                            for (const pan of pg.panels || []) {
                                if (pan.panel_id === panelId) {
                                    pan.image_hash = response.content_hash;
                                    pan._last_prompt = response.prompt_used;
                                }
                            }
                        }
                    }

                    this._hidePanelSpinner(panelId);
                } catch (error) {
                    console.error(`Failed to regenerate ${panelId}:`, error);
                    this._hidePanelSpinner(panelId);
                }
            }

            this._renderPage();
            this._renderPanelSelection();
            btn.textContent = 'Done!';
            setTimeout(() => {
                btn.textContent = 'Regenerate';
                btn.disabled = false;
            }, 2000);
        } catch (error) {
            console.error('Regenerate all failed:', error);
            btn.textContent = 'Regenerate';
            btn.disabled = false;
        }
    }

    async _savePanelToBank() {
        const panel = this.selectedPanel;
        if (!panel || !panel.image_hash) return;

        const characterIds = Object.keys(panel.scripts || {});
        for (const characterId of characterIds) {
            try {
                await api._post(`/api/characters/${characterId}/references`, {
                    content_hash: panel.image_hash,
                    source: 'generated',
                });
                // Reload character so the reference appears in the bank
                await this._reloadCharacter(characterId);
            } catch (error) {
                console.error(`Failed to add to ${characterId} bank:`, error);
            }
        }

        const names = characterIds.map(id => this.characters[id]?.name || id).join(', ');
        console.log(`Saved panel to bank for: ${names}`);
    }

    // --- Character screen ---

    _showCharacterScreen() {
        document.getElementById('splash').classList.add('hidden');
        document.getElementById('chapter-select').classList.add('hidden');
        document.getElementById('app').classList.add('hidden');
        document.getElementById('character-screen').classList.remove('hidden');
        document.getElementById('char-create-from-image').hidden = true;
        this._detailCharacterId = null;
        this._selectedRefHash = null;
        this._selectedSoloPanelId = null;
        this._renderCharacterScreenList();
        // Refresh model indicator
        this._renderModelSelector().catch(() => {});
    }

    async _newCharacterFromImage(file) {
        // Show the creation form immediately with loading state
        const createPanel = document.getElementById('char-create-from-image');
        const detailEmpty = document.getElementById('char-detail-empty');
        const detailContent = document.getElementById('char-detail-content');

        createPanel.hidden = false;
        detailEmpty.hidden = true;
        detailContent.hidden = true;

        // Show image preview
        const previewUrl = URL.createObjectURL(file);
        document.getElementById('char-create-preview').src = previewUrl;
        document.getElementById('char-create-caption').textContent = 'Analyzing image...';

        // Clear all fields
        for (const id of [
            'char-create-name', 'char-create-description',
            'char-create-species', 'char-create-body-type',
            'char-create-hair-colour', 'char-create-hair-style',
            'char-create-eye-colour', 'char-create-skin-tone',
            'char-create-facial-features', 'char-create-outfit',
            'char-create-accessories', 'char-create-art-style',
            'char-create-line-style', 'char-create-rendering',
            'char-create-genre',
        ]) {
            document.getElementById(id).value = '';
        }

        try {
            // Upload and analyze
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('/api/analyze-image', {
                method: 'POST', body: formData,
            });
            if (!response.ok) throw new Error(await response.text());
            const result = await response.json();

            this._newCharImageAnalysis = result;

            // Populate fields from analysis
            document.getElementById('char-create-caption').textContent = result.raw_caption;

            const c = result.character;
            document.getElementById('char-create-species').value = c.species || '';
            document.getElementById('char-create-body-type').value = c.body_type || '';
            document.getElementById('char-create-hair-colour').value = c.hair_colour || '';
            document.getElementById('char-create-hair-style').value = c.hair_style || '';
            document.getElementById('char-create-eye-colour').value = c.eye_colour || '';
            document.getElementById('char-create-skin-tone').value = c.skin_tone || '';
            document.getElementById('char-create-facial-features').value = c.facial_features || '';
            document.getElementById('char-create-outfit').value = c.outfit || '';
            document.getElementById('char-create-accessories').value = c.accessories || '';

            const a = result.art_style;
            document.getElementById('char-create-art-style').value = a.art_style || '';
            document.getElementById('char-create-line-style').value = a.line_style || '';
            document.getElementById('char-create-rendering').value = a.rendering || '';
            document.getElementById('char-create-genre').value = a.genre_hints || '';

            // Focus the name field
            document.getElementById('char-create-name').focus();

        } catch (error) {
            console.error('Image analysis failed:', error);
            document.getElementById('char-create-caption').textContent =
                'Analysis failed — is Ollama running with LLaVA? You can still fill in fields manually.';
        }
    }

    async _confirmCharacterFromImage() {
        const name = document.getElementById('char-create-name').value.trim();
        if (!name) {
            document.getElementById('char-create-name').focus();
            return;
        }

        const btn = document.getElementById('btn-char-create-confirm');
        btn.disabled = true;
        btn.textContent = 'Creating...';

        try {
            // Build appearance prompt from fields
            const fields = {
                species: document.getElementById('char-create-species').value.trim(),
                body_type: document.getElementById('char-create-body-type').value.trim(),
                hair_colour: document.getElementById('char-create-hair-colour').value.trim(),
                hair_style: document.getElementById('char-create-hair-style').value.trim(),
                eye_colour: document.getElementById('char-create-eye-colour').value.trim(),
                skin_tone: document.getElementById('char-create-skin-tone').value.trim(),
                facial_features: document.getElementById('char-create-facial-features').value.trim(),
                outfit: document.getElementById('char-create-outfit').value.trim(),
                accessories: document.getElementById('char-create-accessories').value.trim(),
            };

            // Create the character
            const description = document.getElementById('char-create-description').value.trim();
            const character = await api.createCharacter({ name, description });

            // Apply appearance properties
            await api._put(
                `/api/characters/${character.character_id}/appearance`,
                fields,
            );

            // Add the image to reference bank and accept it
            if (this._newCharImageAnalysis?.content_hash) {
                const hash = this._newCharImageAnalysis.content_hash;
                await api._post(
                    `/api/characters/${character.character_id}/references`,
                    {
                        content_hash: hash,
                        source: 'upload',
                        caption: this._newCharImageAnalysis.raw_caption || '',
                        pose: this._newCharImageAnalysis.character?.pose || '',
                        expression: this._newCharImageAnalysis.character?.expression || '',
                    }
                );
                // Accept the reference
                await api._put(
                    `/api/characters/${character.character_id}/references/${hash}`,
                    { accepted: true }
                );
            }

            // Set story art style if empty
            const artStyle = document.getElementById('char-create-art-style').value.trim();
            const genre = document.getElementById('char-create-genre').value.trim();
            if (artStyle || genre) {
                const storyUpdate = {};
                if (artStyle) storyUpdate.art_style = artStyle;
                if (genre) storyUpdate.genre = genre;
                await api.put('/api/story', storyUpdate);
                if (this.storyData) {
                    if (artStyle) this.storyData.art_style = artStyle;
                    if (genre) this.storyData.genre = genre;
                }
            }

            // Reload and select the new character
            const allChars = await api.listCharacters();
            this.characters = allChars;
            // Also reload chapters since solo chapter was created
            const storyData = await api.getStory();
            this.chapters = Object.values(storyData.chapters || {});

            this._newCharImageAnalysis = null;
            document.getElementById('char-create-from-image').hidden = true;
            await this._selectCharacterInScreen(character.character_id);

        } catch (error) {
            console.error('Failed to create character from image:', error);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Create Character';
        }
    }

    _cancelCharacterFromImage() {
        document.getElementById('char-create-from-image').hidden = true;
        document.getElementById('char-detail-empty').hidden =
            this._detailCharacterId != null;
        document.getElementById('char-detail-content').hidden =
            this._detailCharacterId == null;
        this._newCharImageAnalysis = null;
    }

    _renderCharacterScreenList() {
        const container = document.getElementById('char-list-items');
        container.innerHTML = '';

        for (const [characterId, character] of Object.entries(this.characters)) {
            const item = document.createElement('div');
            item.className = 'char-list-item';
            if (characterId === this._detailCharacterId) item.classList.add('selected');

            const name = document.createElement('div');
            name.textContent = character.name;
            item.appendChild(name);

            if (character.appearance_prompt) {
                const appearance = document.createElement('div');
                appearance.className = 'char-list-appearance';
                appearance.textContent = character.appearance_prompt;
                item.appendChild(appearance);
            }

            item.addEventListener('click', () => this._selectCharacterInScreen(characterId));
            container.appendChild(item);
        }

        if (Object.keys(this.characters).length === 0) {
            container.innerHTML = '<span class="hint">No characters yet</span>';
        }
    }

    async _selectCharacterInScreen(characterId) {
        this._detailCharacterId = characterId;
        this._selectedRefHash = null;
        this._selectedSoloPanelId = null;
        await this._reloadCharacter(characterId);

        // Load solo chapter data
        try {
            this._soloChapter = await api._get(`/api/characters/${characterId}/solo-chapter`);
        } catch (error) {
            console.error('Failed to load solo chapter:', error);
            this._soloChapter = null;
        }

        document.getElementById('char-detail-empty').hidden = true;
        document.getElementById('char-detail-content').hidden = false;
        this._renderCharacterDetail();
        this._renderCharacterScreenList();
    }

    _renderCharacterDetail() {
        const character = this.characters[this._detailCharacterId];
        if (!character) return;

        document.getElementById('character-detail-name').textContent = character.name;

        // Editable appearance properties
        const propsContainer = document.getElementById('character-detail-props');
        propsContainer.innerHTML = '';
        const appearance = character.appearance || {};
        const props = appearance.properties || {};

        const appearanceFields = [
            { key: 'species', label: 'Species' },
            { key: 'body_type', label: 'Body type' },
            { key: 'skin_tone', label: 'Skin tone' },
            { key: 'hair_colour', label: 'Hair colour' },
            { key: 'hair_style', label: 'Hair style' },
            { key: 'eye_colour', label: 'Eye colour' },
            { key: 'facial_features', label: 'Facial features' },
            { key: 'outfit', label: 'Default outfit' },
            { key: 'accessories', label: 'Accessories' },
            { key: 'art_style_notes', label: 'Style notes' },
        ];

        // Character-level negative prompt (separate from appearance properties)
        const negRow = document.createElement('div');
        negRow.className = 'appearance-field-row';
        const negLabel = document.createElement('label');
        negLabel.className = 'appearance-label';
        negLabel.textContent = 'Negative';
        negRow.appendChild(negLabel);
        const negInput = document.createElement('input');
        negInput.type = 'text';
        negInput.className = 'appearance-input';
        negInput.value = character.negative_prompt || '';
        negInput.placeholder = 'Always exclude (e.g. animal ears, tail, fur)...';
        negInput.addEventListener('change', async () => {
            try {
                await api.updateCharacter(this._detailCharacterId, {
                    negative_prompt: negInput.value,
                });
                character.negative_prompt = negInput.value;
            } catch (e) { console.error('Failed to save negative:', e); }
        });
        negRow.appendChild(negInput);
        propsContainer.appendChild(negRow);

        for (const field of appearanceFields) {
            const row = document.createElement('div');
            row.className = 'appearance-field-row';

            const label = document.createElement('label');
            label.className = 'appearance-label';
            label.textContent = field.label;
            row.appendChild(label);

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'appearance-input';
            input.value = props[field.key] || '';
            input.placeholder = field.label + '...';
            input.dataset.field = field.key;
            input.addEventListener('change', () => this._saveAppearanceField(field.key, input.value));
            row.appendChild(input);

            propsContainer.appendChild(row);
        }

        // Solo panels
        this._renderSoloPanelsGrid();

        // Reference grid
        this._renderReferenceGrid();

        // Hide editor
        document.getElementById('reference-editor').hidden = true;
        document.getElementById('solo-panel-feedback').hidden = true;
    }

    async _saveAppearanceField(fieldName, value) {
        if (!this._detailCharacterId) return;
        try {
            await api._put(
                `/api/characters/${this._detailCharacterId}/appearance`,
                { [fieldName]: value }
            );
            // Update local state
            const character = this.characters[this._detailCharacterId];
            if (character?.appearance?.properties) {
                character.appearance.properties[fieldName] = value;
            }
        } catch (error) {
            console.error('Failed to save appearance field:', error);
        }
    }

    _renderSoloPanelsGrid() {
        const grid = document.getElementById('solo-panels-grid');
        grid.innerHTML = '';

        if (!this._soloChapter || !this._soloChapter.pages) {
            grid.innerHTML = '<span class="hint">Generate solo images to build the character sheet</span>';
            return;
        }

        let hasPanels = false;
        for (const page of this._soloChapter.pages) {
            for (const panel of page.panels || []) {
                if (!panel.image_hash) continue;
                hasPanels = true;

                const thumb = document.createElement('div');
                thumb.className = 'solo-panel-thumb';
                if (panel.panel_id === this._selectedSoloPanelId) {
                    thumb.classList.add('selected');
                }

                const img = document.createElement('img');
                img.src = api.contentUrl(panel.image_hash);
                img.alt = 'Solo panel';
                thumb.appendChild(img);

                // Show feedback indicator
                if (panel._feedback === true) {
                    const indicator = document.createElement('span');
                    indicator.className = 'panel-feedback-indicator';
                    indicator.textContent = '\u{1F44D}';
                    thumb.appendChild(indicator);
                } else if (panel._feedback === false) {
                    const indicator = document.createElement('span');
                    indicator.className = 'panel-feedback-indicator';
                    indicator.textContent = '\u{1F44E}';
                    thumb.appendChild(indicator);
                }

                thumb.addEventListener('click', () => this._selectSoloPanel(panel));
                grid.appendChild(thumb);
            }
        }

        if (!hasPanels) {
            grid.innerHTML = '<span class="hint">Generate solo images to build the character sheet</span>';
        }
    }

    _selectSoloPanel(panel) {
        this._selectedSoloPanelId = panel.panel_id;
        this._selectedSoloPanel = panel;

        // Show feedback controls
        document.getElementById('solo-panel-feedback').hidden = false;

        // Update grid selection
        this._renderSoloPanelsGrid();
    }

    async _soloFeedback(accepted) {
        if (!this._selectedSoloPanel?.image_hash) return;
        try {
            const prompt = this._selectedSoloPanel.scripts
                ? Object.values(this._selectedSoloPanel.scripts)[0]?.action || ''
                : '';
            await api.submitFeedback(
                this._selectedSoloPanel.image_hash,
                prompt,
                accepted,
                [this._detailCharacterId],
                this._selectedSoloPanel.panel_id,
            );
            this._selectedSoloPanel._feedback = accepted;
            this._renderSoloPanelsGrid();
        } catch (error) {
            console.error('Solo feedback failed:', error);
        }
    }

    async _soloRegenerate() {
        if (!this._selectedSoloPanel) return;
        const script = this._selectedSoloPanel.scripts
            ? Object.values(this._selectedSoloPanel.scripts)[0]
            : null;
        const prompt = script?.action || '';
        if (!prompt) return;

        try {
            const result = await api.generatePanel(this._selectedSoloPanel.panel_id);
            this._selectedSoloPanel.image_hash = result.content_hash;
            this._renderSoloPanelsGrid();
        } catch (error) {
            console.error('Solo regenerate failed:', error);
        }
    }

    async _openCharacterDetail(characterId) {
        // Redirect to character screen
        this._showCharacterScreen();
        await this._selectCharacterInScreen(characterId);
    }

    _renderReferenceGrid() {
        const character = this.characters[this._detailCharacterId];
        const grid = document.getElementById('reference-grid');
        grid.innerHTML = '';

        const refs = character?.appearance?.references || [];
        if (refs.length === 0) {
            grid.innerHTML = '<span class="hint">No references yet — upload images to build the training bank</span>';
            return;
        }

        for (const ref of refs) {
            const thumb = document.createElement('div');
            thumb.className = 'reference-thumb';
            if (ref.accepted === true) thumb.classList.add('accepted');
            else if (ref.accepted === false) thumb.classList.add('rejected');
            else thumb.classList.add('unrated');

            const img = document.createElement('img');
            img.src = api.contentUrl(ref.content_hash);
            img.alt = ref.caption || 'Reference';
            thumb.appendChild(img);

            const source = document.createElement('span');
            source.className = 'ref-source';
            source.textContent = ref.source;
            thumb.appendChild(source);

            thumb.addEventListener('click', () => this._selectReference(ref));
            grid.appendChild(thumb);
        }
    }

    _selectReference(ref) {
        this._selectedRefHash = ref.content_hash;
        const editor = document.getElementById('reference-editor');
        editor.hidden = false;

        document.getElementById('reference-preview').src = api.contentUrl(ref.content_hash);
        document.getElementById('ref-caption').value = ref.caption || '';
        document.getElementById('ref-pose').value = ref.pose || '';
        document.getElementById('ref-expression').value = ref.expression || '';
        document.getElementById('ref-angle').value = ref.angle || '';
        document.getElementById('ref-scene').value = ref.scene || '';
        document.getElementById('ref-outfit').value = ref.outfit_variant || '';
    }

    async _saveReferenceLabels() {
        if (!this._detailCharacterId || !this._selectedRefHash) return;
        try {
            await api._put(
                `/api/characters/${this._detailCharacterId}/references/${this._selectedRefHash}`,
                {
                    caption: document.getElementById('ref-caption').value,
                    pose: document.getElementById('ref-pose').value,
                    expression: document.getElementById('ref-expression').value,
                    angle: document.getElementById('ref-angle').value,
                    scene: document.getElementById('ref-scene').value,
                    outfit_variant: document.getElementById('ref-outfit').value,
                }
            );
            // Reload character data
            await this._reloadCharacter(this._detailCharacterId);
            this._renderReferenceGrid();
        } catch (error) {
            console.error('Failed to save labels:', error);
        }
    }

    async _rateReference(accepted) {
        if (!this._detailCharacterId || !this._selectedRefHash) return;
        try {
            await api._put(
                `/api/characters/${this._detailCharacterId}/references/${this._selectedRefHash}`,
                { accepted }
            );
            await this._reloadCharacter(this._detailCharacterId);
            this._renderReferenceGrid();
        } catch (error) {
            console.error('Failed to rate reference:', error);
        }
    }

    async _deleteReference() {
        if (!this._detailCharacterId || !this._selectedRefHash) return;
        try {
            await api._delete(
                `/api/characters/${this._detailCharacterId}/references/${this._selectedRefHash}`
            );
            await this._reloadCharacter(this._detailCharacterId);
            this._selectedRefHash = null;
            document.getElementById('reference-editor').hidden = true;
            this._renderReferenceGrid();
        } catch (error) {
            console.error('Failed to delete reference:', error);
        }
    }

    async _generateSolo() {
        if (!this._detailCharacterId) return;

        const prompt = document.getElementById('solo-prompt').value.trim();
        const pose = document.getElementById('solo-pose').value.trim();
        const emotion = document.getElementById('solo-emotion').value.trim();
        const outfit = document.getElementById('solo-outfit').value.trim();
        const direction = document.getElementById('solo-direction').value.trim();
        const shotType = document.getElementById('solo-shot-type').value;
        const negative = document.getElementById('solo-negative').value.trim();
        const seedVal = document.getElementById('solo-seed').value;

        // Need at least one visual field
        if (!prompt && !pose && !emotion && !outfit && !direction) {
            document.getElementById('solo-prompt').focus();
            return;
        }

        const goBtn = document.getElementById('btn-solo-go');
        goBtn.disabled = true;
        goBtn.textContent = 'Generating...';

        try {
            const body = {
                character_id: this._detailCharacterId,
                prompt, pose, outfit, emotion, direction,
                shot_type: shotType,
            };
            if (negative) body.negative_prompt = negative;
            if (seedVal) body.seed = parseInt(seedVal, 10);

            const result = await api._post('/api/generate-solo', body);

            await this._reloadCharacter(this._detailCharacterId);

            // Reload solo chapter to show the new panel
            try {
                this._soloChapter = await api._get(
                    `/api/characters/${this._detailCharacterId}/solo-chapter`
                );
            } catch (e) { /* ignore */ }

            this._renderSoloPanelsGrid();
            this._renderReferenceGrid();

            // Clear fields
            document.getElementById('solo-prompt').value = '';
            document.getElementById('solo-pose').value = '';
            document.getElementById('solo-emotion').value = '';
            document.getElementById('solo-outfit').value = '';
            document.getElementById('solo-direction').value = '';
            document.getElementById('solo-shot-type').value = '';
            document.getElementById('solo-negative').value = '';
            document.getElementById('solo-seed').value = '';
            document.getElementById('solo-generate-area').hidden = true;
        } catch (error) {
            console.error('Solo generation failed:', error);
        } finally {
            goBtn.disabled = false;
            goBtn.textContent = 'Generate';
        }
    }

    async _uploadReferences(files) {
        if (!this._detailCharacterId) return;
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch(
                    `/api/characters/${this._detailCharacterId}/references/upload`,
                    { method: 'POST', body: formData }
                );
                if (!response.ok) throw new Error(await response.text());
            } catch (error) {
                console.error('Failed to upload reference:', error);
            }
        }
        await this._reloadCharacter(this._detailCharacterId);
        this._renderReferenceGrid();
    }

    async _uploadAndAnalyze(file) {
        if (!this._detailCharacterId) return;

        const analysisEl = document.getElementById('analysis-result');
        const captionEl = document.getElementById('analysis-caption');
        const charFieldsEl = document.getElementById('analysis-character-fields');
        const artFieldsEl = document.getElementById('analysis-art-style-fields');

        // Show loading state
        analysisEl.hidden = false;
        captionEl.textContent = 'Analyzing image...';
        charFieldsEl.innerHTML = '';
        artFieldsEl.innerHTML = '';

        try {
            // Upload and analyze in one step
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('/api/analyze-image', {
                method: 'POST', body: formData,
            });
            if (!response.ok) throw new Error(await response.text());
            const result = await response.json();

            // Also add to the character's reference bank
            await api._post(
                `/api/characters/${this._detailCharacterId}/references`,
                { content_hash: result.content_hash, source: 'upload' }
            );
            await this._reloadCharacter(this._detailCharacterId);
            this._renderReferenceGrid();

            // Store for Apply
            this._pendingAnalysis = result;

            // Display results
            captionEl.textContent = result.raw_caption;

            // Character fields
            charFieldsEl.innerHTML = '<h5>Character</h5>';
            for (const [key, value] of Object.entries(result.character)) {
                if (!value) continue;
                const row = document.createElement('div');
                row.className = 'analysis-field';
                row.innerHTML = `<strong>${key.replace(/_/g, ' ')}:</strong> ${value}`;
                charFieldsEl.appendChild(row);
            }

            // Art style fields
            artFieldsEl.innerHTML = '<h5>Art Style</h5>';
            for (const [key, value] of Object.entries(result.art_style)) {
                if (!value) continue;
                const row = document.createElement('div');
                row.className = 'analysis-field';
                row.innerHTML = `<strong>${key.replace(/_/g, ' ')}:</strong> ${value}`;
                artFieldsEl.appendChild(row);
            }

        } catch (error) {
            console.error('Upload and analyze failed:', error);
            captionEl.textContent = 'Analysis failed — is Ollama running with LLaVA?';
        }
    }

    async _applyAnalysis() {
        if (!this._detailCharacterId || !this._pendingAnalysis) return;

        const applyBtn = document.getElementById('btn-apply-analysis');
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';

        try {
            const result = await api._post(
                `/api/characters/${this._detailCharacterId}/apply-analysis`,
                {
                    character: this._pendingAnalysis.character,
                    art_style: this._pendingAnalysis.art_style,
                }
            );

            await this._reloadCharacter(this._detailCharacterId);
            this._renderCharacterDetail();

            // Update story settings display if art style was set
            if (this.storyData) {
                this.storyData.art_style = result.story_art_style || this.storyData.art_style;
                this.storyData.genre = result.story_genre || this.storyData.genre;
            }

            document.getElementById('analysis-result').hidden = true;
            this._pendingAnalysis = null;

        } catch (error) {
            console.error('Failed to apply analysis:', error);
        } finally {
            applyBtn.disabled = false;
            applyBtn.textContent = 'Apply to Character';
        }
    }

    async _reloadCharacter(characterId) {
        try {
            const characters = await api.listCharacters();
            if (characters[characterId]) {
                this.characters[characterId] = characters[characterId];
            }
        } catch (error) {
            console.error('Failed to reload character:', error);
        }
    }

    // --- Character manager ---

    _openCharacterManager() {
        document.getElementById('character-manager-overlay').classList.add('visible');
        this._renderCharacterList();
    }

    _closeCharacterManager() {
        document.getElementById('character-manager-overlay').classList.remove('visible');
        // Refresh character screen if visible
        if (!document.getElementById('character-screen').classList.contains('hidden')) {
            this._renderCharacterScreenList();
        } else {
            this._renderSidebar();
        }
    }

    _renderCharacterList() {
        const container = document.getElementById('character-list-full');
        container.innerHTML = '';

        for (const character of Object.values(this.characters)) {
            const row = document.createElement('div');
            row.className = 'character-row';

            const name = document.createElement('span');
            name.className = 'character-row-name';
            name.textContent = character.name;
            if (character.is_temporary) {
                const tag = document.createElement('span');
                tag.className = 'temporary-tag';
                tag.textContent = 'temp';
                name.appendChild(tag);
            }
            row.appendChild(name);

            const appearance = document.createElement('span');
            appearance.className = 'character-row-detail';
            appearance.textContent = character.appearance_prompt || '(no appearance set)';
            row.appendChild(appearance);

            const actions = document.createElement('div');
            actions.className = 'character-row-actions';

            const editBtn = document.createElement('button');
            editBtn.textContent = 'Edit';
            editBtn.addEventListener('click', () => this._editCharacter(character.character_id));
            actions.appendChild(editBtn);

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.addEventListener('click', () => this._deleteCharacter(character.character_id));
            actions.appendChild(deleteBtn);

            row.appendChild(actions);
            container.appendChild(row);
        }
    }

    _editCharacter(characterId) {
        const character = this.characters[characterId];
        if (!character) return;

        this._editingCharacterId = characterId;
        document.getElementById('edit-char-name').value = character.name || '';
        document.getElementById('edit-char-appearance').value = character.appearance_prompt || '';
        document.getElementById('edit-char-personality').value = character.personality_prompt || '';
        document.getElementById('edit-char-description').value = character.description || '';
        document.getElementById('edit-char-temporary').checked = character.is_temporary || false;
        document.getElementById('add-character-form').hidden = true;
        document.getElementById('edit-character-form').hidden = false;
    }

    async _saveCharacterEdit() {
        if (!this._editingCharacterId) return;

        const data = {
            name: document.getElementById('edit-char-name').value.trim() || null,
            appearance_prompt: document.getElementById('edit-char-appearance').value.trim() || null,
            personality_prompt: document.getElementById('edit-char-personality').value.trim() || null,
            description: document.getElementById('edit-char-description').value.trim() || null,
        };

        try {
            const updated = await api.updateCharacter(this._editingCharacterId, data);
            this.characters[this._editingCharacterId] = updated;
            this._cancelCharacterEdit();
            this._renderCharacterList();
        } catch (error) {
            console.error('Failed to update character:', error);
        }
    }

    _cancelCharacterEdit() {
        this._editingCharacterId = null;
        document.getElementById('add-character-form').hidden = false;
        document.getElementById('edit-character-form').hidden = true;
    }

    async _deleteCharacter(characterId) {
        try {
            await api.deleteCharacter(characterId);
            delete this.characters[characterId];
            this._renderCharacterList();
        } catch (error) {
            console.error('Failed to delete character:', error);
        }
    }

    async _createCharacter() {
        const nameInput = document.getElementById('new-char-name');
        const name = nameInput.value.trim();
        if (!name) { nameInput.focus(); return; }

        try {
            // 1. Create character in story
            const character = await api.createCharacter({
                name: name,
                appearance_prompt: document.getElementById('new-char-appearance').value.trim(),
                personality_prompt: document.getElementById('new-char-personality').value.trim(),
                description: document.getElementById('new-char-description').value.trim(),
                is_temporary: document.getElementById('new-char-temporary').checked,
            });
            this.characters[character.character_id] = character;

            // 2. Add to current chapter — characters enter through the chapter
            const chapter = this.currentChapter;
            if (chapter) {
                const updatedChapter = await api.addCharacterToChapter(
                    chapter.chapter_id, character.character_id
                );
                chapter.character_ids = updatedChapter.character_ids;
            }

            nameInput.value = '';
            document.getElementById('new-char-appearance').value = '';
            document.getElementById('new-char-personality').value = '';
            document.getElementById('new-char-description').value = '';
            document.getElementById('new-char-temporary').checked = false;
            this._renderCharacterList();
        } catch (error) {
            console.error('Failed to create character:', error);
        }
    }

    // --- Splash screen / story management ---

    async _showSplash() {
        document.getElementById('app').classList.add('hidden');
        document.getElementById('chapter-select').classList.add('hidden');
        document.getElementById('character-screen').classList.add('hidden');
        document.getElementById('splash').classList.remove('hidden');
        await this._loadSavedStoriesList();
    }

    async _loadSavedStoriesList() {
        const list = document.getElementById('saved-stories-list');
        list.innerHTML = '';

        try {
            const stories = await api.listSavedStories();
            if (stories.length === 0) {
                list.innerHTML = '<span class="saved-story-meta">No saved stories yet</span>';
                return;
            }

            for (const story of stories) {
                const item = document.createElement('div');
                item.className = 'saved-story-item';

                const name = document.createElement('span');
                name.className = 'saved-story-name';
                name.textContent = story.filename.replace('.cvn', '').replace(/_/g, ' ');
                item.appendChild(name);

                const meta = document.createElement('span');
                meta.className = 'saved-story-meta';
                const size = (story.size_bytes / 1024).toFixed(0);
                meta.textContent = `${size} KB`;
                item.appendChild(meta);

                item.addEventListener('click', () => this._loadSavedStory(story.filename));
                list.appendChild(item);
            }
        } catch (error) {
            list.innerHTML = '<span class="saved-story-meta">Could not load stories</span>';
        }
    }

    async _newStoryAndEnter() {
        try {
            await api.newStory('Untitled Story');
            await this._enterStory();
        } catch (error) {
            console.error('Failed to create new story:', error);
        }
    }

    async _loadSavedStory(filename) {
        try {
            await api.loadSavedStory(filename);
            await this._enterStory();
        } catch (error) {
            console.error('Failed to load story:', error);
        }
    }

    async _loadStoryFile(file) {
        try {
            await api.uploadStory(file);
            await this._enterStory();
        } catch (error) {
            console.error('Failed to load story file:', error);
        }
    }

    async _saveStory() {
        try {
            const result = await api.saveStory();
            console.log('Story saved:', result.filename);
        } catch (error) {
            console.error('Failed to save story:', error);
        }
    }

    _downloadStory() {
        window.location.href = api.downloadStoryUrl();
    }

    async _importCharacterFile(file) {
        try {
            const result = await api.importCharacter(file);
            console.log('Imported characters:', result.count);
            // Reload character data
            const storyData = await api.getStory();
            this._loadStoryData(storyData);
            this._renderCharacterList();
            this._renderSidebar();
        } catch (error) {
            console.error('Failed to import characters:', error);
        }
    }
}


// Boot
document.addEventListener('DOMContentLoaded', () => {
    window.comiventure = new ComiventureApp();
});
