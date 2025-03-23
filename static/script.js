// Global variables for pagination
let currentDocumentPage = 1;
let currentNotePage = 1;

// Load data when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.documentElement.classList.add('dark');
    }
    
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

// Dark mode toggle function
function toggleDarkMode() {
    if (document.documentElement.classList.contains('dark')) {
        document.documentElement.classList.remove('dark');
        localStorage.setItem('darkMode', 'disabled');
    } else {
        document.documentElement.classList.add('dark');
        localStorage.setItem('darkMode', 'enabled');
    }
}

// Tab switching functionality
function openTab(tabId) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.add('hidden');
        tab.classList.remove('block');
    });
    
    // Reset all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('bg-white', 'dark:bg-gray-800');
        button.classList.add('bg-gray-100', 'dark:bg-gray-700');
    });
    
    // Show the selected tab content
    const selectedTab = document.getElementById(tabId);
    selectedTab.classList.remove('hidden');
    selectedTab.classList.add('block');
    
    // Style the active tab button
    const activeButton = Array.from(tabButtons).find(button => 
        button.getAttribute('onclick').includes(tabId)
    );
    if (activeButton) {
        activeButton.classList.remove('bg-gray-100', 'dark:bg-gray-700');
        activeButton.classList.add('bg-white', 'dark:bg-gray-800');
    }
}

function search() {
    const query = document.getElementById('search-input').value;
    console.log("Searching for:", query);
    
    if (!query) {
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.innerHTML = '<div class="p-4 text-gray-600 dark:text-gray-400">Enter a search query</div>';
        resultsDiv.classList.remove('hidden');
        return;
    }

    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        console.log("Search results:", data);
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.classList.remove('hidden');
        
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<div class="p-4 text-gray-600 dark:text-gray-400">No results found</div>';
            return;
        }
        
        resultsDiv.innerHTML = data.results.map(r => {
            if (r.type === 'task') {
                // For tasks, link to switch to the Kanban tab and focus on the task
                return `<div class="p-4 border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <div class="flex items-center gap-2 mb-1">
                        <svg class="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                        </svg>
                        <a href="#" onclick="viewTask(${r.id})" class="font-medium text-blue-600 dark:text-blue-400 hover:underline">${r.title}</a>
                        <span class="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">Task</span>
                    </div>
                    <p class="text-sm text-gray-600 dark:text-gray-300 mb-2">${r.description}</p>
                    <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>Added on: ${r.date}</span>
                        <span>Match: ${(r.similarity * 100).toFixed(1)}%</span>
                    </div>
                </div>`;
            } else {
                // For documents with links (external resources)
                if (r.link) {
                    return `<div class="p-4 border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                        <div class="flex items-center gap-2 mb-1">
                            <svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path>
                            </svg>
                            <a href="#" onclick="viewDocument(${r.id})" class="font-medium text-blue-600 dark:text-blue-400 hover:underline">${r.title}</a>
                            <span class="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Link</span>
                        </div>
                        <a href="${r.link}" target="_blank" class="text-sm text-gray-600 dark:text-gray-300 hover:underline block mb-2">${r.link}</a>
                        <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>Added on: ${r.date}</span>
                            <span>Match: ${(r.similarity * 100).toFixed(1)}%</span>
                        </div>
                    </div>`;
                } 
                // For notes (documents without links)
                else {
                    return `<div class="p-4 border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                        <div class="flex items-center gap-2 mb-1">
                            <svg class="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path>
                            </svg>
                            <a href="#" onclick="viewNote(${r.id})" class="font-medium text-blue-600 dark:text-blue-400 hover:underline">${r.title}</a>
                            <span class="px-2 py-0.5 text-xs rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">Note</span>
                        </div>
                        <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>Added on: ${r.date}</span>
                            <span>Match: ${(r.similarity * 100).toFixed(1)}%</span>
                        </div>
                    </div>`;
                }
            }
        }).join('');
    })
    .catch(error => {
        console.error('Error searching:', error);
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.classList.remove('hidden');
        resultsDiv.innerHTML = '<div class="p-4 text-red-600 dark:text-red-400">Error performing search</div>';
    });
}

// Functions to view specific items from search results
function viewTask(id) {
    // Switch to Kanban tab
    openTab('kanban-tab');
    
    // Fetch the task info to get its status
    fetch(`/get_task?id=${id}`)
    .then(response => response.json())
    .then(task => {
        // Wait for the tab switch and tasks to load
        setTimeout(() => {
            // Find the task in the column
            const taskElement = document.querySelector(`.task[data-id="${id}"]`);
            if (taskElement) {
                // Scroll to the task
                taskElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Highlight the task briefly
                taskElement.classList.add('highlight');
                setTimeout(() => taskElement.classList.remove('highlight'), 2000);
            }
        }, 300);
    })
    .catch(error => {
        console.error('Error fetching task for highlighting:', error);
    });
    
    // Hide search results
    document.getElementById('search-results').classList.add('hidden');
}

function viewDocument(id) {
    // Switch to Documents tab
    openTab('documents-tab');
    
    // Load the document list if needed
    loadDocuments(1);
    
    // Find and highlight the document (in a real app, you might need pagination handling)
    setTimeout(() => {
        highlightItemInList('documents-container', id);
    }, 300);
    
    // Hide search results
    document.getElementById('search-results').classList.add('hidden');
}

function viewNote(id) {
    // Switch to Notes tab
    openTab('notes-tab');
    
    // Load the notes list if needed
    loadNotes(1);
    
    // Find and highlight the note (in a real app, you might need pagination handling)
    setTimeout(() => {
        highlightItemInList('notes-container', id);
    }, 300);
    
    // Hide search results
    document.getElementById('search-results').classList.add('hidden');
}

function highlightItemInList(containerId, itemId) {
    // This is a simple implementation that would need to be enhanced
    // for real pagination, but demonstrates the concept
    const container = document.getElementById(containerId);
    const items = container.querySelectorAll('.item-card');
    
    // For each item, check if it's the one we're looking for
    let found = false;
    items.forEach(item => {
        // Parse data attributes or look for ID in the DOM
        // This depends on how the IDs are stored in your item cards
        const cardButtons = item.querySelectorAll('.action-button');
        cardButtons.forEach(button => {
            const onclickAttr = button.getAttribute('onclick');
            if (onclickAttr && onclickAttr.includes(`(${itemId})`)) {
                // This is our item
                found = true;
                
                // Scroll to it
                item.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Highlight it
                item.classList.add('highlight');
                setTimeout(() => item.classList.remove('highlight'), 2000);
            }
        });
    });
    
    // If not found on this page, in a real app you would need to handle
    // searching through pagination
    if (!found) {
        console.log(`Item ${itemId} not found on current page`);
    }
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
    if (type === 'document') {
        document.getElementById('edit-link-container').classList.remove('hidden');
    } else {
        document.getElementById('edit-link-container').classList.add('hidden');
    }
    
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
        
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    })
    .catch(error => {
        console.error('Error fetching document details:', error);
        alert('Error fetching document details');
    });
}

function closeModal() {
    document.getElementById('edit-modal').classList.add('hidden');
    document.getElementById('edit-modal').classList.remove('flex');
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
        document.getElementById('todo').innerHTML = '<h2 class="text-lg font-semibold text-center pb-3 border-b dark:border-gray-700 mb-4">Todo</h2>';
        document.getElementById('doing').innerHTML = '<h2 class="text-lg font-semibold text-center pb-3 border-b dark:border-gray-700 mb-4">Doing</h2>';
        document.getElementById('done').innerHTML = '<h2 class="text-lg font-semibold text-center pb-3 border-b dark:border-gray-700 mb-4">Done</h2>';
        
        // Add tasks to columns
        for (const task of data.tasks) {
            const taskElement = document.createElement('div');
            taskElement.className = 'bg-white dark:bg-gray-700 rounded-lg shadow-sm p-4 mb-3 cursor-grab transition-all duration-200 hover:shadow-md';
            taskElement.draggable = true;
            taskElement.dataset.id = task.id;
            taskElement.dataset.status = task.status;
            
            // Create status badge based on status
            let statusBadge = '';
            if (task.status === 'todo') {
                statusBadge = '<span class="inline-block px-2 py-1 text-xs rounded-full bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">Todo</span>';
            } else if (task.status === 'doing') {
                statusBadge = '<span class="inline-block px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">Doing</span>';
            } else if (task.status === 'done') {
                statusBadge = '<span class="inline-block px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Done</span>';
            }
            
            taskElement.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-medium text-gray-900 dark:text-white">${task.title}</h3>
                    ${statusBadge}
                </div>
                <p class="text-gray-600 dark:text-gray-300 text-sm mb-3">${task.description}</p>
                <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500 dark:text-gray-400">${task.date}</span>
                    <div class="flex gap-2">
                        <button onclick="openTaskEditModal(${task.id})" class="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                        </button>
                        <button onclick="openTaskDeleteModal(${task.id})" class="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                        </button>
                    </div>
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
        
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    })
    .catch(error => {
        console.error('Error fetching task details:', error);
        alert('Error fetching task details');
    });
}

function closeTaskModal() {
    document.getElementById('edit-task-modal').classList.add('hidden');
    document.getElementById('edit-task-modal').classList.remove('flex');
}

function openTaskDeleteModal(id) {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('delete-id').value = id;
    document.getElementById('delete-type').value = 'task';
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function openDeleteModal(id) {
    const modal = document.getElementById('confirm-modal');
    document.getElementById('delete-id').value = id;
    document.getElementById('delete-type').value = 'document';
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeConfirmModal() {
    document.getElementById('confirm-modal').classList.add('hidden');
    document.getElementById('confirm-modal').classList.remove('flex');
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
    
    if (searchResults && searchBar) {
        if (!searchResults.contains(event.target) && !searchBar.contains(event.target)) {
            searchResults.classList.add('hidden');
        }
    }
});