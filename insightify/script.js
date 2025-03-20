const sidebar = document.getElementById('sidebar');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');

function toggleMenu() {
    sidebar.classList.toggle('open');
}

fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    for (const file of files) {
        addFileToSidebar(file);
    }
});

function addFileToSidebar(file) {
    const extension = file.name.split('.').pop().toLowerCase();
    const icon = getIcon(extension);

    const fileItem = document.createElement('div');
    fileItem.classList.add('file-item');

    fileItem.innerHTML = `
        <div class="left">
            <i class="fa ${icon}"></i>
            <span>${file.name}</span>
        </div>
        <div>
            <span class="file-extension">${extension}</span>
            <button class="delete-btn" onclick="deleteFile(this)">×</button>
        </div>
    `;

    fileList.appendChild(fileItem);
}

function deleteFile(button) {
    const item = button.closest('.file-item');
    item.remove();
}

function getIcon(ext) {
    switch(ext) {
        case 'csv': return 'fa-file-csv';
        case 'xlsx': return 'fa-file-excel';
        case 'pdf': return 'fa-file-pdf';
        case 'txt': return 'fa-file-lines';
        default: return 'fa-file';
    }
}
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    if (userInput.value.trim() !== '') {
        // User message
        const userMessage = document.createElement('div');
        userMessage.className = 'message-bubble user-message';
        userMessage.textContent = userInput.value;
        
        // Bot message
        const botMessage = document.createElement('div');
        botMessage.className = 'message-bubble bot-message';
        botMessage.textContent = "Okay then, sweet dreams everyone...";

        chatMessages.appendChild(userMessage);
        chatMessages.appendChild(botMessage);

        userInput.value = '';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// home
function startNewChat() {
    window.location.href = "chat.html"; 
}

function showLogin() {
    document.getElementById('loginModal').style.display = 'block';
}

function hideLogin() {
    document.getElementById('loginModal').style.display = 'none';
}