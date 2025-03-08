// Global variables for pagination
let currentDocumentPage = 1;
let currentNotePage = 1;

// Load data when the page loads
document.addEventListener('DOMContentLoaded', function() {
    loadTasks();
    loadDocuments(1);
    loadNotes(1);
    
    // Add event listener for Enter key in search input
    document.getElementById('search-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            search();
        }
    });
    
    // Load documents and notes when their tabs are clicked
    document.querySelector('button[onclick="openTab(\'documents-tab\')"]').addEventListener('click', function() {
        loadDocuments(currentDocumentPage);
    });
    
    document.querySelector('button[onclick="openTab(\'notes-tab\')"]').addEventListener('click', function() {
        loadNotes(currentNotePage);
    });
});

// Tab switching functionality
function openTab(tabId) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => button.classList.remove('active'));
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Add active class to the clicked button
    const activeButton = Array.from(tabButtons).find(button => 
        button.getAttribute('onclick').includes(tabId)
    );
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

function search() {
    const query = document.getElementById('search-input').value;
    if (!query) {
        document.getElementById('search-results').innerHTML = '<div>Enter a search query</div>';
        document.getElementById('search-results').classList.add('active');
        return;
    }

    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.classList.add('active');
        
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<div>No results found</div>';
            return;
        }
        
        resultsDiv.innerHTML = data.results.map(r => {
            if (r.type === 'task') {
                return `<div>
                    <strong>${r.title}</strong> (${r.date}) - Task
                    <p>${r.description}</p>
                    <small>Similarity: ${(r.similarity * 100).toFixed(1)}%</small>
                </div>`;
            } else {
                return `<div>
                    <strong>${r.title}</strong> (${r.date}) - Document
                    ${r.link ? `<p><a href="${r.link}" target="_blank">${r.link}</a></p>` : '<p>Note</p>'}
                    <small>Similarity: ${(r.similarity * 100).toFixed(1)}%</small>
                </div>`;
            }
        }).join('');
    })
    .catch(error => {
        console.error('Error searching:', error);
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.classList.add('active');
        resultsDiv.innerHTML = '<div>Error performing search</div>';
    });
}

function addNote() {
    const title = document.getElementById('note-title').value;
    const content = document.getElementById('note-content').value;
    
    if (!title || !content) {
        alert('Please enter both title and content');
        return;
    }
    
    fetch('/add_note', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, content })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        document.getElementById('note-title').value = '';
        document.getElementById('note-content').value = '';
        loadNotes(1); // Reload the first page of notes
    })
    .catch(error => {
        console.error('Error adding note:', error);
        alert('Error adding note');
    });
}

function loadNotes(page) {
    if (page < 1) page = 1;
    currentNotePage = page;
    
    fetch(`/get_notes?page=${page}&per_page=5`)
    .then(response => response.json())
    .then(data => {
        const notesContainer = document.getElementById('notes-container');
        notesContainer.innerHTML = '';
        
        if (data.notes.length === 0) {
            notesContainer.innerHTML = '<p>No notes found</p>';
        } else {
            data.notes.forEach(note => {
                const noteElement = document.createElement('div');
                noteElement.className = 'item-card';
                // Truncate content if it's too long
                const truncatedContent = note.content.length > 150 
                    ? note.content.substring(0, 150) + '...' 
                    : note.content;
                
                noteElement.innerHTML = `
                    <h3>${note.title}</h3>
                    <p>${truncatedContent}</p>
                    <p class="date">Added on: ${note.date}</p>
                    <div class="item-actions">
                        <button class="action-button edit-button" onclick="openEditModal(${note.id}, 'note')">Edit</button>
                        <button class="action-button delete-button" onclick="openDeleteModal(${note.id})">Delete</button>
                    </div>
                `;
                notesContainer.appendChild(noteElement);
            });
        }
        
        // Update pagination controls
        document.getElementById('note-page-info').textContent = `Page ${data.pagination.current_page} of ${data.pagination.total_pages}`;
        document.getElementById('prev-notes').disabled = !data.pagination.has_prev;
        document.getElementById('next-notes').disabled = !data.pagination.has_next;
    })
    .catch(error => {
        console.error('Error loading notes:', error);
        document.getElementById('notes-container').innerHTML = '<p>Error loading notes</p>';
    });
}

function addLink() {
    const title = document.getElementById('link-title').value;
    const link = document.getElementById('link-url').value;
    
    if (!title || !link) {
        alert('Please enter both title and URL');
        return;
    }
    
    // Basic URL validation
    if (!link.startsWith('http://') && !link.startsWith('https://')) {
        alert('URL must start with http:// or https://');
        return;
    }
    
    fetch('/add_link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, link })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        document.getElementById('link-title').value = '';
        document.getElementById('link-url').value = '';
        loadDocuments(1); // Reload the first page of documents
    })
    .catch(error => {
        console.error('Error adding link:', error);
        alert('Error adding link');
    });
}

function loadDocuments(page) {
    if (page < 1) page = 1;
    currentDocumentPage = page;
    
    fetch(`/get_documents?page=${page}&per_page=5`)
    .then(response => response.json())
    .then(data => {
        const documentsContainer = document.getElementById('documents-container');
        documentsContainer.innerHTML = '';
        
        if (data.documents.length === 0) {
            documentsContainer.innerHTML = '<p>No documents found</p>';
        } else {
            data.documents.forEach(doc => {
                const docElement = document.createElement('div');
                docElement.className = 'item-card';
                docElement.innerHTML = `
                    <h3>${doc.title}</h3>
                    <p><a href="${doc.link}" target="_blank">${doc.link}</a></p>
                    <p class="date">Added on: ${doc.date}</p>
                    <div class="item-actions">
                        <button class="action-button edit-button" onclick="openEditModal(${doc.id}, 'document')">Edit</button>
                        <button class="action-button delete-button" onclick="openDeleteModal(${doc.id})">Delete</button>
                    </div>
                `;
                documentsContainer.appendChild(docElement);
            });
        }
        
        // Update pagination controls
        document.getElementById('document-page-info').textContent = `Page ${data.pagination.current_page} of ${data.pagination.total_pages}`;
        document.getElementById('prev-documents').disabled = !data.pagination.has_prev;
        document.getElementById('next-documents').disabled = !data.pagination.has_next;
    })
    .catch(error => {
        console.error('Error loading documents:', error);
        document.getElementById('documents-container').innerHTML = '<p>Error loading documents</p>';
    });
}

// Modal functions
function openEditModal(id, type) {
    const modal = document.getElementById('edit-modal');
    document.getElementById('edit-id').value = id;
    document.getElementById('edit-type').value = type;
    
    // Update modal title
    document.getElementById('modal-title').textContent = type === 'document' ? 'Edit Document' : 'Edit Note';
    
    // Show/hide link field based on type
    document.getElementById('edit-link-container').style.display = type === 'document' ? 'block' : 'none';
    
    // Fetch the current data
    fetch(`/get_document?id=${id}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById('edit-title').value = data.title;
        
        if (type === 'document') {
            document.getElementById('edit-link').value = data.link || '';
            document.getElementById('edit-content').value = '';
        } else {
            document.getElementById('edit-content').value = data.content || '';
        }
        
        modal.style.display = 'block';
    })
    .catch(error => {
        console.error('Error fetching document details:', error);
        alert('Error fetching document details');
    });
}

function closeModal() {
    document.getElementById('edit-modal').style.display = 'none';
}

function saveEdit() {
    const id = document.getElementById('edit-id').value;
    const type = document.getElementById('edit-type').value;
    const title = document.getElementById('edit-title').value;
    
    if (!title) {
        alert('Title is required');
        return;
    }
    
    const data = {
        id: parseInt(id),
        title: title
    };
    
    if (type === 'document') {
        data.link = document.getElementById('edit-link').value;
    } else {
        data.content = document.getElementById('edit-content').value;
    }
    
    fetch('/update_document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        alert(result.message);
        closeModal();
        
        // Reload the appropriate list
        if (type === 'document') {
            loadDocuments(currentDocumentPage);
        } else {
            loadNotes(currentNotePage);
        }
    })
    .catch(error => {
        console.error('Error updating document:', error);
        alert('Error updating document');
    });
}

function openDeleteModal(id) {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('delete-id').value = id;
    modal.style.display = 'block';
}

function closeConfirmModal() {
    document.getElementById('confirm-modal').style.display = 'none';
}

function confirmDelete() {
    const id = document.getElementById('delete-id').value;
    const type = document.getElementById('delete-type').value;
    
    let endpoint = '/delete_document';
    
    if (type === 'task') {
        endpoint = '/delete_task';
    }
    
    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: parseInt(id) })
    })
    .then(response => response.json())
    .then(result => {
        alert(result.message);
        closeConfirmModal();
        
        if (type === 'task') {
            loadTasks();
        } else {
            // Reload both document and note lists
            loadDocuments(currentDocumentPage);
            loadNotes(currentNotePage);
        }
    })
    .catch(error => {
        console.error(`Error deleting ${type}:`, error);
        alert(`Error deleting ${type}`);
    });
}

function uploadFile() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    if (!file.name.endsWith('.txt')) {
        alert('Only .txt files are allowed');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload_txt', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        document.getElementById('file-input').value = '';
        loadNotes(1); // Reload the first page of notes since uploaded files are stored as notes
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
    });
}

function addTask() {
    const title = document.getElementById('task-title').value;
    const description = document.getElementById('task-description').value;
    const status = document.getElementById('task-status').value;
    
    if (!title || !description) {
        alert('Please enter both title and description');
        return;
    }
    
    fetch('/add_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, description, status })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        document.getElementById('task-title').value = '';
        document.getElementById('task-description').value = '';
        document.getElementById('task-status').value = 'todo';
        loadTasks(); // Reload tasks to show the new one
    })
    .catch(error => {
        console.error('Error adding task:', error);
        alert('Error adding task');
    });
}

function loadTasks() {
    fetch('/get_tasks')
    .then(response => response.json())
    .then(data => {
        // Clear existing tasks
        document.getElementById('todo').innerHTML = '<h2>Todo</h2>';
        document.getElementById('doing').innerHTML = '<h2>Doing</h2>';
        document.getElementById('done').innerHTML = '<h2>Done</h2>';
        
        // Add tasks to columns
        for (const task of data.tasks) {
            const taskElement = document.createElement('div');
            taskElement.className = 'task';
            taskElement.draggable = true;
            taskElement.dataset.id = task.id;
            taskElement.dataset.status = task.status;
            
            taskElement.innerHTML = `
                <h3>${task.title}</h3>
                <p>${task.description}</p>
                <div class="task-date">${task.date}</div>
                <div class="task-actions">
                    <button class="action-button edit-button" onclick="openTaskEditModal(${task.id})">Edit</button>
                    <button class="action-button delete-button" onclick="openTaskDeleteModal(${task.id})">Delete</button>
                </div>
            `;
            
            // Add drag and drop event listeners
            taskElement.addEventListener('dragstart', handleDragStart);
            taskElement.addEventListener('dragend', handleDragEnd);
            
            document.getElementById(task.status).appendChild(taskElement);
        }
        
        // Set up drop zones
        setupDropZones();
    })
    .catch(error => {
        console.error('Error loading tasks:', error);
    });
}

// Task editing functions
function openTaskEditModal(id) {
    const modal = document.getElementById('edit-task-modal');
    document.getElementById('edit-task-id').value = id;
    
    // Fetch the current task data
    fetch(`/get_task?id=${id}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById('edit-task-title').value = data.title;
        document.getElementById('edit-task-description').value = data.description;
        document.getElementById('edit-task-status').value = data.status;
        
        modal.style.display = 'block';
    })
    .catch(error => {
        console.error('Error fetching task details:', error);
        alert('Error fetching task details');
    });
}

function closeTaskModal() {
    document.getElementById('edit-task-modal').style.display = 'none';
}

function saveTaskEdit() {
    const id = document.getElementById('edit-task-id').value;
    const title = document.getElementById('edit-task-title').value;
    const description = document.getElementById('edit-task-description').value;
    const status = document.getElementById('edit-task-status').value;
    
    if (!title || !description) {
        alert('Title and description are required');
        return;
    }
    
    fetch('/update_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            id: parseInt(id),
            title: title,
            description: description,
            status: status
        })
    })
    .then(response => response.json())
    .then(result => {
        alert(result.message);
        closeTaskModal();
        loadTasks();
    })
    .catch(error => {
        console.error('Error updating task:', error);
        alert('Error updating task');
    });
}

function openTaskDeleteModal(id) {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('delete-id').value = id;
    document.getElementById('delete-type').value = 'task';
    modal.style.display = 'block';
}

// Drag and drop functionality
let draggedTask = null;

function handleDragStart(e) {
    this.classList.add('dragging');
    draggedTask = this;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.id);
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    draggedTask = null;
    
    // Remove drop-target styling from all columns
    document.querySelectorAll('.column').forEach(column => {
        column.classList.remove('drop-target');
    });
}

function setupDropZones() {
    const columns = document.querySelectorAll('.column');
    
    columns.forEach(column => {
        column.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('drop-target');
        });
        
        column.addEventListener('dragleave', function(e) {
            this.classList.remove('drop-target');
        });
        
        column.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('drop-target');
            
            const taskId = e.dataTransfer.getData('text/plain');
            const newStatus = this.id; // column id is the status
            
            // Only update if status changed
            if (draggedTask && draggedTask.dataset.status !== newStatus) {
                updateTaskStatus(taskId, newStatus);
            }
        });
    });
}

function updateTaskStatus(taskId, newStatus) {
    fetch('/update_task_status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            id: parseInt(taskId),
            status: newStatus
        })
    })
    .then(response => response.json())
    .then(result => {
        loadTasks(); // Reload all tasks to reflect the new order
    })
    .catch(error => {
        console.error('Error updating task status:', error);
        alert('Error updating task status');
    });
}

// Hide search results when clicking outside
document.addEventListener('click', function(event) {
    const searchResults = document.getElementById('search-results');
    const searchBar = document.querySelector('.search-bar');
    
    if (!searchResults.contains(event.target) && !searchBar.contains(event.target)) {
        searchResults.classList.remove('active');
    }
});