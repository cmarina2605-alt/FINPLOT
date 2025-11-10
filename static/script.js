// DOM elements
const chat = document.getElementById('chat');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const ingestBtn = document.getElementById('ingestBtn');
const pdfInput = document.getElementById('pdfInput');
const ingestStatus = document.getElementById('ingestStatus');

/**
 * Add a message bubble to the chat
 * @param {string} text - Message text
 * @param {string} type - 'user' or 'bot'
 * @param {Array} citations - Array of citation objects {source, page}
 */
function addMessage(text, type, citations = []) {
    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    msg.innerHTML = text.replace(/\n/g, '<br>'); // preserve line breaks

    // Add citations if present
    if (citations.length > 0) {
        const cit = document.createElement('div');
        cit.className = 'citation';
        cit.innerHTML = 'Sources: ' + citations.map(c => `${c.source} (p.${c.page})`).join(', ');
        msg.appendChild(cit);
    }

    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
}

// Send question when clicking the button
sendBtn.onclick = async () => {
    const question = questionInput.value.trim();
    if (!question) return;

    // Show user message
    addMessage(question, 'user');
    questionInput.value = '';

    // Temporary "thinking" message
    addMessage('Thinking...', 'bot');

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                k: 5,
                model: modelSelect.value   // selected model from dropdown
            })
        });

        const data = await response.json();

        // Remove "thinking" message
        chat.removeChild(chat.lastChild);

        if (data.answer) {
            addMessage(data.answer, 'bot', data.citations || []);
        } else {
            addMessage('Error: ' + (data.detail || 'Unknown error'), 'bot');
        }
    } catch (err) {
        chat.removeChild(chat.lastChild);
        addMessage('Connection error. Is Ollama running?', 'bot');
    }
};

// Allow Enter (without Shift) to send
questionInput.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
    }
});

// Upload PDFs when clicking the Load button
ingestBtn.onclick = async () => {
    const files = pdfInput.files;
    if (files.length === 0) {
        ingestStatus.textContent = 'Please select at least one PDF';
        return;
    }

    ingestStatus.textContent = 'Uploading...';

    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/ingest-file', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();
            ingestStatus.textContent = result.result || result.detail || 'Uploaded';
        } catch (err) {
            ingestStatus.textContent = 'Upload failed';
        }
    }

    // Clear status after 3 seconds
    setTimeout(() => ingestStatus.textContent = '', 3000);
};

// Welcome message when page loads
addMessage('Hello! I am FinPilot. Load your PDFs and ask me anything.', 'bot');