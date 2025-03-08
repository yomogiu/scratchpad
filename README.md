# Scratchpad

This app lets the users add notes, tasks (Kanban board), track links, and upload txt files and make everything searchable. It uses a vector-based semantic search (via embeddings) to return the most likely results.

Using all-MiniLM-L6-v2 so sometimes the result can be janky, but it's decently fast. You can swap with another model of your liking by updating app.py.

https://github.com/user-attachments/assets/42d5c145-20cf-4ac7-ac55-0491c308cff2.webm



## Features

- **Document Storage**:
  - Take notes
  - Save links to external resources
  - Upload text files
  
- **Semantic Search**:
  - Vector embeddings using Sentence Transformers
  - Cosine similarity for matching content (may add dot product as another option)
  - Search across documents, notes, and tasks
  
- **Task Management**:
  - Kanban board with Todo, Doing, and Done columns
  - Add tasks with title, description, and status
  - Tasks are also searchable via embeddings

## Technical Stack

- **Backend**: Flask
- **Database**: SQLite
- **ML**: Sentence Transformers for embeddings
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd scratchpad_v2
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Access the application in your browser:
   ```
   http://localhost:5000
   ```

3. Features:
   - Use the search bar at the top to find documents and tasks
   - Add tasks using the "Add Task" form
   - Add notes using the "Add Note" form
   - Add links using the "Add Link" form
   - Upload text files using the "Upload Text File" section

## Project Structure

```
scratchpad_v2/
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
├── docs.db              # SQLite database (created on first run)
├── static/
│   ├── index.html       # Main HTML page
│   ├── style.css        # CSS styles
│   └── script.js        # JavaScript functionality
└── README.md            # This file
```
