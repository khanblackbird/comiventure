/**
 * Panel editor — inline mask drawing tool for AI inpainting.
 * Overlays a canvas directly on the selected panel element.
 * No separate overlay — edit right where the image is.
 */
class PanelEditor {
    constructor() {
        this.isEditing = false;
        this.isDrawing = false;
        this.tool = 'brush';
        this.brushSize = 20;
        this.currentPanelId = null;
        this.canvas = null;
        this.context = null;
        this.sourceImage = null;
        this.panelElement = null;
        this.toolbar = null;
        this.onApplyEdit = null;
        // No DOM needed at construction — canvas is created inline when editing
    }

    open(panelData) {
        if (this.isEditing) this.close();

        this.currentPanelId = panelData.panel_id;
        const imageSource = panelData.image_url || panelData.image_path;
        if (!imageSource) return;

        // Find the panel DOM element
        this.panelElement = document.querySelector(
            `.comic-panel[data-panel-id="${panelData.panel_id}"]`
        );
        if (!this.panelElement) return;

        // Load the source image to get dimensions
        this.sourceImage = new Image();
        this.sourceImage.crossOrigin = 'anonymous';
        this.sourceImage.onload = () => this._setupCanvas();
        this.sourceImage.src = imageSource;
    }

    _setupCanvas() {
        // Store native image dimensions for mask extraction
        this.nativeWidth = this.sourceImage.width;
        this.nativeHeight = this.sourceImage.height;

        // Figure out how the image actually displays with object-fit: contain
        const panelRect = this.panelElement.getBoundingClientRect();
        const imageAspect = this.nativeWidth / this.nativeHeight;
        const panelAspect = panelRect.width / panelRect.height;

        let displayWidth, displayHeight, offsetX, offsetY;
        if (imageAspect > panelAspect) {
            // Image is wider — fits width, letterboxed top/bottom
            displayWidth = panelRect.width;
            displayHeight = panelRect.width / imageAspect;
            offsetX = 0;
            offsetY = (panelRect.height - displayHeight) / 2;
        } else {
            // Image is taller — fits height, pillarboxed left/right
            displayHeight = panelRect.height;
            displayWidth = panelRect.height * imageAspect;
            offsetX = (panelRect.width - displayWidth) / 2;
            offsetY = 0;
        }

        this.displayOffset = { x: offsetX, y: offsetY };
        this.displaySize = { width: displayWidth, height: displayHeight };

        // Canvas matches the image's displayed area exactly
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'edit-canvas-overlay';
        this.canvas.width = Math.round(displayWidth);
        this.canvas.height = Math.round(displayHeight);
        this.canvas.style.left = `${offsetX}px`;
        this.canvas.style.top = `${offsetY}px`;
        this.context = this.canvas.getContext('2d');

        // Draw source image to fill the canvas (no distortion)
        this.context.drawImage(this.sourceImage, 0, 0, this.canvas.width, this.canvas.height);

        // Insert canvas over the panel
        this.panelElement.classList.add('editing');
        this.panelElement.appendChild(this.canvas);

        // Create inline toolbar
        this._createToolbar();

        // Bind draw events
        this.canvas.addEventListener('mousedown', (event) => this._startDraw(event));
        this.canvas.addEventListener('mousemove', (event) => this._draw(event));
        this.canvas.addEventListener('mouseup', () => this._stopDraw());
        this.canvas.addEventListener('mouseleave', () => this._stopDraw());

        this.isEditing = true;
    }

    _createToolbar() {
        this.toolbar = document.createElement('div');
        this.toolbar.className = 'edit-toolbar';
        this.toolbar.innerHTML = `
            <button class="edit-tool-btn active" data-tool="brush">Brush</button>
            <button class="edit-tool-btn" data-tool="eraser">Eraser</button>
            <label class="edit-size-label">Size:
                <input type="range" class="edit-size-slider" min="5" max="80" value="${this.brushSize}">
            </label>
            <label class="edit-size-label">Strength:
                <input type="range" class="edit-strength-slider" min="50" max="100" value="85">
                <span class="edit-strength-value">0.85</span>
            </label>
            <input type="text" class="edit-prompt-input" placeholder="Describe what should be here (not what to remove)...">
            <button class="edit-apply-btn">Apply</button>
            <button class="edit-cancel-btn">Cancel</button>
        `;

        this.panelElement.appendChild(this.toolbar);

        // Tool buttons
        for (const toolButton of this.toolbar.querySelectorAll('.edit-tool-btn')) {
            toolButton.addEventListener('click', () => {
                this.tool = toolButton.dataset.tool;
                for (const button of this.toolbar.querySelectorAll('.edit-tool-btn')) {
                    button.classList.toggle('active', button === toolButton);
                }
            });
        }

        // Size slider
        this.toolbar.querySelector('.edit-size-slider').addEventListener('input', (event) => {
            this.brushSize = parseInt(event.target.value);
        });

        // Strength slider
        const strengthSlider = this.toolbar.querySelector('.edit-strength-slider');
        const strengthValue = this.toolbar.querySelector('.edit-strength-value');
        strengthSlider.addEventListener('input', () => {
            strengthValue.textContent = (parseInt(strengthSlider.value) / 100).toFixed(2);
        });

        // Apply / Cancel
        this.toolbar.querySelector('.edit-apply-btn').addEventListener('click', () => this._applyEdit());
        this.toolbar.querySelector('.edit-cancel-btn').addEventListener('click', () => this.close());

        // Enter to apply
        this.toolbar.querySelector('.edit-prompt-input').addEventListener('keydown', (event) => {
            if (event.key === 'Enter') this._applyEdit();
        });
    }

    close() {
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
        if (this.toolbar && this.toolbar.parentNode) {
            this.toolbar.parentNode.removeChild(this.toolbar);
        }
        if (this.panelElement) {
            this.panelElement.classList.remove('editing');
        }

        this.canvas = null;
        this.context = null;
        this.toolbar = null;
        this.panelElement = null;
        this.sourceImage = null;
        this.currentPanelId = null;
        this.isEditing = false;
        this.isDrawing = false;
    }

    _startDraw(event) {
        this.isDrawing = true;
        this._draw(event);
    }

    _draw(event) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const cursorX = event.clientX - rect.left;
        const cursorY = event.clientY - rect.top;

        this.context.beginPath();
        this.context.arc(cursorX, cursorY, this.brushSize, 0, Math.PI * 2);

        if (this.tool === 'brush') {
            this.context.fillStyle = 'rgba(255, 0, 0, 0.4)';
            this.context.fill();
        } else {
            if (this.sourceImage) {
                this.context.save();
                this.context.clip();
                this.context.drawImage(
                    this.sourceImage, 0, 0, this.canvas.width, this.canvas.height
                );
                this.context.restore();
            }
        }
    }

    _stopDraw() {
        this.isDrawing = false;
    }

    _getMaskBase64() {
        // Extract mask at display resolution
        const displayMask = document.createElement('canvas');
        displayMask.width = this.canvas.width;
        displayMask.height = this.canvas.height;
        const displayCtx = displayMask.getContext('2d');

        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const maskImageData = displayCtx.createImageData(this.canvas.width, this.canvas.height);

        for (let pixelIndex = 0; pixelIndex < imageData.data.length; pixelIndex += 4) {
            const red = imageData.data[pixelIndex];
            const green = imageData.data[pixelIndex + 1];
            const isMasked = red > 100 && green < 100;
            const maskValue = isMasked ? 255 : 0;
            maskImageData.data[pixelIndex] = maskValue;
            maskImageData.data[pixelIndex + 1] = maskValue;
            maskImageData.data[pixelIndex + 2] = maskValue;
            maskImageData.data[pixelIndex + 3] = 255;
        }
        displayCtx.putImageData(maskImageData, 0, 0);

        // Scale mask to native image resolution for the model
        const nativeMask = document.createElement('canvas');
        nativeMask.width = this.nativeWidth;
        nativeMask.height = this.nativeHeight;
        const nativeCtx = nativeMask.getContext('2d');
        nativeCtx.imageSmoothingEnabled = false;
        nativeCtx.drawImage(displayMask, 0, 0, this.nativeWidth, this.nativeHeight);

        return nativeMask.toDataURL('image/png').split(',')[1];
    }

    async _applyEdit() {
        if (!this.currentPanelId || !this.toolbar) return;

        const promptInput = this.toolbar.querySelector('.edit-prompt-input');
        const prompt = promptInput.value.trim();
        if (!prompt) {
            promptInput.focus();
            return;
        }

        // Extract mask, panel ID, and strength before closing
        const maskBase64 = this._getMaskBase64();
        const panelId = this.currentPanelId;
        const strengthSlider = this.toolbar.querySelector('.edit-strength-slider');
        const strength = parseInt(strengthSlider.value) / 100;

        // Close editor first so spinner appears on a clean panel
        this.close();

        // Then run inpainting — spinner shows during this
        if (this.onApplyEdit) {
            await this.onApplyEdit(panelId, maskBase64, prompt, strength);
        }
    }
}
