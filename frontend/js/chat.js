/**
 * Chat panel controller — handles player input, character selection, dialogue display.
 */
class ChatController {
    constructor(historyElement, inputElement, sendButton, characterSelectElement, choiceArea) {
        this.historyElement = historyElement;
        this.inputElement = inputElement;
        this.sendButton = sendButton;
        this.characterSelectElement = characterSelectElement;
        this.choiceArea = choiceArea;
        this.selectedCharacterId = null;
        this.onSendMessage = null;
        this.onChoiceSelected = null;

        this._bindEvents();
    }

    _bindEvents() {
        this.sendButton.addEventListener('click', () => this._handleSend());
        this.inputElement.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this._handleSend();
            }
        });
    }

    _handleSend() {
        const message = this.inputElement.value.trim();
        if (!message || !this.selectedCharacterId) return;

        this.addMessage('player', 'You', message);
        this.inputElement.value = '';

        if (this.onSendMessage) {
            this.onSendMessage(this.selectedCharacterId, message);
        }
    }

    addMessage(role, senderName, text) {
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${role}`;

        const sender = document.createElement('div');
        sender.className = 'sender';
        sender.textContent = senderName;
        messageElement.appendChild(sender);

        const content = document.createElement('div');
        content.textContent = text;
        messageElement.appendChild(content);

        this.historyElement.appendChild(messageElement);
        this.historyElement.scrollTop = this.historyElement.scrollHeight;
    }

    renderCharacters(characters) {
        this.characterSelectElement.innerHTML = '';

        for (const character of characters) {
            const portrait = document.createElement('div');
            portrait.className = 'character-portrait placeholder';
            portrait.textContent = character.name.charAt(0);
            portrait.title = character.name;
            portrait.dataset.characterId = character.character_id;

            if (character.portrait_path) {
                const img = document.createElement('img');
                img.src = character.portrait_path;
                img.className = 'character-portrait';
                img.title = character.name;
                img.dataset.characterId = character.character_id;
                portrait.replaceWith(img);
                this.characterSelectElement.appendChild(img);
            } else {
                this.characterSelectElement.appendChild(portrait);
            }
        }

        this.characterSelectElement.addEventListener('click', (event) => {
            const target = event.target.closest('[data-character-id]');
            if (!target) return;
            this._selectCharacter(target.dataset.characterId);
        });
    }

    _selectCharacter(characterId) {
        this.selectedCharacterId = characterId;
        const allPortraits = this.characterSelectElement.querySelectorAll('[data-character-id]');
        for (const portrait of allPortraits) {
            portrait.classList.toggle('active', portrait.dataset.characterId === characterId);
        }
    }

    showChoices(choices) {
        this.choiceArea.innerHTML = '';
        this.choiceArea.hidden = false;

        for (const choice of choices) {
            const button = document.createElement('button');
            button.className = 'choice-button';
            button.textContent = choice.text;
            button.addEventListener('click', () => {
                this.choiceArea.hidden = true;
                if (this.onChoiceSelected) {
                    this.onChoiceSelected(choice);
                }
            });
            this.choiceArea.appendChild(button);
        }
    }
}
