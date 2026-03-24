/**
 * Panel editor — mask drawing tool for AI inpainting edits.
 * User draws over a panel image to create a mask, enters a prompt,
 * and the backend regenerates the masked region.
 */
class PanelEditor {
    constructor(overlayElement, canvasElement, promptInput, brushSizeInput) {
        this.overlay = overlayElement;
        this.canvas = canvasElement;
        this.context = canvasElement.getContext('2d');
        this.promptInput = promptInput;
        this.brushSizeInput = brushSizeInput;

        this.isDrawing = false;
        this.tool = 'brush';  // 'brush' or 'eraser'
        this.currentPanelData = null;
        this.sourceImage = null;
        this.onApplyEdit = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.canvas.addEventListener('mousedown', (event) => this._startDraw(event));
        this.canvas.addEventListener('mousemove', (event) => this._draw(event));
        this.canvas.addEventListener('mouseup', () => this._stopDraw());
        this.canvas.addEventListener('mouseleave', () => this._stopDraw());

        document.getElementById('btn-brush').addEventListener('click', () => this._setTool('brush'));
        document.getElementById('btn-eraser').addEventListener('click', () => this._setTool('eraser'));
        document.getElementById('btn-apply-edit').addEventListener('click', () => this._applyEdit());
        document.getElementById('btn-cancel-edit').addEventListener('click', () => this.close());
    }

    open(panelData) {
        this.currentPanelData = panelData;
        this.promptInput.value = '';
        this.overlay.hidden = false;

        if (panelData.image_path) {
            this.sourceImage = new Image();
            this.sourceImage.onload = () => {
                this.canvas.width = this.sourceImage.width;
                this.canvas.height = this.sourceImage.height;
                this.context.drawImage(this.sourceImage, 0, 0);
            };
            this.sourceImage.src = panelData.image_path;
        } else {
            this.canvas.width = 768;
            this.canvas.height = 512;
            this.context.fillStyle = '#f0f0f0';
            this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    close() {
        this.overlay.hidden = true;
        this.currentPanelData = null;
        this.sourceImage = null;
    }

    _setTool(tool) {
        this.tool = tool;
        document.getElementById('btn-brush').classList.toggle('active', tool === 'brush');
        document.getElementById('btn-eraser').classList.toggle('active', tool === 'eraser');
    }

    _startDraw(event) {
        this.isDrawing = true;
        this._draw(event);
    }

    _draw(event) {
        if (!this.isDrawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const cursorX = (event.clientX - rect.left) * scaleX;
        const cursorY = (event.clientY - rect.top) * scaleY;
        const brushRadius = parseInt(this.brushSizeInput.value);

        this.context.beginPath();
        this.context.arc(cursorX, cursorY, brushRadius, 0, Math.PI * 2);

        if (this.tool === 'brush') {
            this.context.fillStyle = 'rgba(255, 0, 0, 0.4)';
            this.context.fill();
        } else {
            // Eraser: restore the original image in this area
            if (this.sourceImage) {
                this.context.save();
                this.context.clip();
                this.context.drawImage(this.sourceImage, 0, 0);
                this.context.restore();
            }
        }
    }

    _stopDraw() {
        this.isDrawing = false;
    }

    _getMaskDataUrl() {
        /**
         * Extract just the mask as a black/white PNG.
         * Red painted areas become white (edit region), everything else black.
         */
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = this.canvas.width;
        maskCanvas.height = this.canvas.height;
        const maskContext = maskCanvas.getContext('2d');

        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const maskImageData = maskContext.createImageData(this.canvas.width, this.canvas.height);

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

        maskContext.putImageData(maskImageData, 0, 0);
        return maskCanvas.toDataURL('image/png');
    }

    async _applyEdit() {
        if (!this.currentPanelData) return;

        const prompt = this.promptInput.value.trim();
        if (!prompt) {
            this.promptInput.focus();
            return;
        }

        const maskDataUrl = this._getMaskDataUrl();
        const maskBase64 = maskDataUrl.split(',')[1];

        if (this.onApplyEdit) {
            await this.onApplyEdit(this.currentPanelData.panel_id, maskBase64, prompt);
        }

        this.close();
    }
}
