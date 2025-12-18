// Function to update the UI with a simple message
function updateOutput(id, content, isCode = false) {
    const element = document.getElementById(id);
    if (isCode) {
        element.textContent = content; // Use textContent for code (pre tag)
    } else {
        // Escape HTML to prevent XSS, then replace newlines with <br>
        const escaped = escapeHTML(content);
        element.innerHTML = escaped.replace(/\\n/g, '<br>');
    }
}

// Simple HTML escaping utility
function escapeHTML(str) {
    if (!str) return '';
    return str.replace(/[&<>"]/g, function (tag) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;'
        };
        return map[tag] || tag;
    });
}

// Function to handle the full prompt process
async function processPrompt() {
    const promptInput = document.getElementById('prompt-input');
    const submitButton = document.getElementById('submit-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const outputContainer = document.getElementById('output-container');
    const userPromptText = promptInput.value.trim();

    if (userPromptText === "") {
        alert("Please enter a question.");
        return;
    }

    // 1. Initial State Setup
    submitButton.disabled = true;
    loadingIndicator.classList.remove('hidden');
    outputContainer.style.display = 'block';

    // Clear previous results
    updateOutput('user-prompt-output', userPromptText);
    updateOutput('cypher-output', 'Generating...');
    updateOutput('final-answer-output', 'Synthesizing...');

    // Setup abort controller for timeout (10 seconds)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    try {
        const response = await fetch('http://localhost:5000/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: userPromptText }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        updateOutput('cypher-output', data.cypher_query, true);
        updateOutput('final-answer-output', data.final_answer);
    } catch (error) {
        console.error('Error processing request:', error);
        const msg = error.name === 'AbortError' ? 'Request timed out after 10 seconds.' : error.message;
        updateOutput('cypher-output', `Error: ${msg}`, true);
        updateOutput('final-answer-output', `I could not process the request due to an error. Details: ${msg}`);
    } finally {
        // 3. Reset State
        loadingIndicator.classList.add('hidden');
        submitButton.disabled = false;
    }
}