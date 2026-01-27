// app.js
const API_BASE_URL = 'http://localhost:8000/api/v1/forecast';

let messageHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    addAssistantMessage('Hello! 👋 I\'m your Load Forecasting Assistant. I can help you with historical demand data, forecasting, trends, and more. What would you like to know?');
});

// Handle Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendQuery();
    }
}

// Send sample query
function sendSampleQuery(element) {
    const query = element.textContent;
    document.getElementById('queryInput').value = query;
    sendQuery();
}

// Send query to API
async function sendQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Add user message
    addUserMessage(query);
    
    // Clear input
    input.value = '';
    
    // Show loading
    showLoading(true);
    disableInput(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add assistant response
        addAssistantResponse(data);
        
    } catch (error) {
        console.error('Error:', error);
        addAssistantMessage(
            '❌ Sorry, I encountered an error processing your query. Please try again.',
            { error: error.message }
        );
    } finally {
        showLoading(false);
        disableInput(false);
        input.focus();
    }
}

// Add user message to chat
function addUserMessage(text) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div>${escapeHtml(text)}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;
    
    chatArea.appendChild(messageDiv);
    scrollToBottom();
    
    messageHistory.push({ role: 'user', content: text });
}

// Add assistant response
function addAssistantResponse(data) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    let content = `<div>${formatText(data.answer)}</div>`;
    
    // Add SQL query if available
    if (data.sql_query) {
        content += `
            <div class="metadata">
                <strong>📊 SQL Query:</strong><br>
                <code>${escapeHtml(data.sql_query)}</code>
            </div>
        `;
    }
    
    // Add data table if available
    if (data.sample_data && data.sample_data.length > 0) {
        content += createDataTable(data.sample_data, data.row_count);
    }
    
    // Add graph if available
    if (data.has_graph && data.graph_image) {
        content += `
            <div class="graph-container">
                <img src="data:image/png;base64,${data.graph_image}" alt="Data Visualization">
                ${data.graph_data ? `
                    <div class="metadata">
                        <strong>📈 Graph Info:</strong> 
                        ${data.graph_data.plot_type} plot with ${data.graph_data.data_points} data points
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    // Add metadata
    if (data.metadata) {
        const meta = [];
        if (data.intent) meta.push(`Intent: ${data.intent}`);
        if (data.row_count) meta.push(`Rows: ${data.row_count.toLocaleString()}`);
        
        if (meta.length > 0) {
            content += `
                <div class="metadata">
                    <strong>ℹ️ Info:</strong> ${meta.join(' | ')}
                </div>
            `;
        }
    }
    
    // Add error if present
    if (!data.success && data.error) {
        content += `
            <div class="error">
                <strong>Error:</strong> ${escapeHtml(data.error)}
            </div>
        `;
    }
    
    content += `<div class="message-time">${getCurrentTime()}</div>`;
    
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    
    chatArea.appendChild(messageDiv);
    scrollToBottom();
    
    messageHistory.push({ role: 'assistant', content: data.answer });
}

// Add simple assistant message
function addAssistantMessage(text, metadata = null) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    let content = `<div>${formatText(text)}</div>`;
    
    if (metadata) {
        content += `
            <div class="metadata">
                ${JSON.stringify(metadata, null, 2)}
            </div>
        `;
    }
    
    content += `<div class="message-time">${getCurrentTime()}</div>`;
    
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    
    chatArea.appendChild(messageDiv);
    scrollToBottom();
}

// Create data table
function createDataTable(data, totalRows) {
    if (!data || data.length === 0) return '';
    
    const keys = Object.keys(data[0]);
    
    let html = '<div class="data-table">';
    html += '<table>';
    html += '<thead><tr>';
    keys.forEach(key => {
        html += `<th>${escapeHtml(key)}</th>`;
    });
    html += '</tr></thead>';
    html += '<tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        keys.forEach(key => {
            html += `<td>${escapeHtml(String(row[key]))}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    if (totalRows && totalRows > data.length) {
        html += `<div style="margin-top: 10px; font-size: 12px; color: #666;">
            Showing ${data.length} of ${totalRows.toLocaleString()} total rows
        </div>`;
    }
    
    html += '</div>';
    
    return html;
}

// Utility functions
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.add('active');
    } else {
        loading.classList.remove('active');
    }
}

function disableInput(disable) {
    document.getElementById('queryInput').disabled = disable;
    document.getElementById('sendBtn').disabled = disable;
}

function scrollToBottom() {
    const chatArea = document.getElementById('chatArea');
    chatArea.scrollTop = chatArea.scrollHeight;
}

function getCurrentTime() {
    return new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatText(text) {
    // Convert newlines to <br>
    text = escapeHtml(text);
    text = text.replace(/\n/g, '<br>');
    
    // Bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic text
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return text;
}