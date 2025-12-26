const messageInput = document.getElementById('messageInput');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen = document.querySelector('.welcome-screen');
let conversationHistory = [];

/**
 * Auto-resize the textarea
 */
function autoResize(element) {
    element.style.height = 'auto';
    element.style.height = element.scrollHeight + 'px';
}

/**
 * Handle Enter key to send message
 */
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Send message to API
 */
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;

    // UI Updates
    messageInput.value = '';
    messageInput.style.height = 'auto';
    welcomeScreen.classList.add('welcome-hidden');

    // Add user message
    addMessage(text, 'user');

    // Add loading indicator
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                conversationHistory: conversationHistory
            })
        });

        const data = await response.json();

        // Remove loading
        removeMessage(loadingId);

        if (response.ok) {
            addMessage(data.response, 'bot', data.sources);

            // Update history
            conversationHistory.push({ role: 'user', content: text });
            conversationHistory.push({ role: 'assistant', content: data.response });
        } else {
            addMessage(`Error: ${data.detail || 'Failed to get response'}`, 'bot');
        }

    } catch (error) {
        removeMessage(loadingId);
        addMessage(`Connection error: ${error.message}`, 'bot');
    }
}

/**
 * Add message to UI
 */
function addMessage(text, type, sources = []) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.id = `msg-${Date.now()}`;

    const icon = type === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    // Parse markdown if bot
    let content = text;
    if (type === 'bot') {
        content = marked.parse(text);

        // Append sources if present
        if (sources && sources.length > 0) {
            const sourcesHtml = sources.map(s =>
                `<span class="source-badge" title="Score: ${s.score ? s.score.toFixed(2) : 'N/A'}">
                    <i class="fa-solid fa-book"></i> ${s.module || 'Textbook'} - ${s.title || 'Page'}
                 </span>`
            ).join('');

            content += `<div class="sources-list">${sourcesHtml}</div>`;
        }
    } else {
        // Escape HTML for user input to prevent XSS
        content = escapeHtml(text).replace(/\n/g, '<br>');
    }

    div.innerHTML = `
        <div class="message-avatar">
            ${icon}
        </div>
        <div class="message-content">
            ${content}
        </div>
    `;

    messagesContainer.appendChild(div);
    scrollToBottom();
    return div.id;
}

/**
 * Add loading bubbles
 */
function addLoadingIndicator() {
    const id = `loading-${Date.now()}`;
    const div = document.createElement('div');
    div.className = 'message bot';
    div.id = id;

    div.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="message-content" style="display: flex; gap: 4px; padding: 1.25rem;">
            <div class="dot" style="width: 8px; height: 8px; background: #94a3b8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both;"></div>
            <div class="dot" style="width: 8px; height: 8px; background: #94a3b8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; animation-delay: -0.32s;"></div>
            <div class="dot" style="width: 8px; height: 8px; background: #94a3b8; border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; animation-delay: -0.16s;"></div>
        </div>
    `;

    // Add bounce animation keyframes if not present
    if (!document.getElementById('bounce-style')) {
        const style = document.createElement('style');
        style.id = 'bounce-style';
        style.innerHTML = `
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
        `;
        document.head.appendChild(style);
    }

    messagesContainer.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function sendSuggestion(text) {
    messageInput.value = text;
    sendMessage();
}

function startNewChat() {
    conversationHistory = [];
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.classList.remove('welcome-hidden');

    // Reset welcome screen animation
    welcomeScreen.style.opacity = '0';
    setTimeout(() => welcomeScreen.style.opacity = '1', 10);
}

function clearChat() {
    if (confirm('Are you sure you want to clear this conversation?')) {
        startNewChat();
    }
}

/**
 * Helper to escape HTML characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Global exposure for onClick events
window.startNewChat = startNewChat;
window.sendSuggestion = sendSuggestion;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.handleKeyDown = handleKeyDown;
window.autoResize = autoResize;
