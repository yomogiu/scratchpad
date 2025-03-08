from flask import Flask, request, send_from_directory
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import os

app = Flask(__name__, static_folder='static', static_url_path='')
model = SentenceTransformer('all-MiniLM-L6-v2')

def init_db():
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        link TEXT,
        date TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL,
        date TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        doc_id TEXT,
        embedding BLOB
    )''')  # TEXT type to handle task prefixes
    conn.commit()
    conn.close()

def store_embedding(doc_id, content):
    embedding = model.encode(content)
    embedding_bytes = embedding.tobytes()
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("INSERT INTO embeddings (doc_id, embedding) VALUES (?, ?)",
              (doc_id, embedding_bytes))
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/add_link', methods=['POST'])
def add_link():
    data = request.json
    title = data['title']
    link = data['link']
    date = datetime.now().strftime('%Y-%m-%d')
    content = "Linked document"
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (title, content, link, date) VALUES (?, ?, ?, ?)",
              (title, content, link, date))
    doc_id = c.lastrowid
    conn.commit()
    store_embedding(str(doc_id), content)
    conn.close()
    return {"message": "Link added"}

@app.route('/upload_txt', methods=['POST'])
def upload_txt():
    file = request.files['file']
    title = file.filename.rsplit('.', 1)[0]
    content = file.read().decode('utf-8')
    date = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (title, content, link, date) VALUES (?, ?, ?, ?)",
              (title, content, None, date))
    doc_id = c.lastrowid
    conn.commit()
    store_embedding(str(doc_id), content)
    conn.close()
    return {"message": "File uploaded"}

@app.route('/add_note', methods=['POST'])
def add_note():
    data = request.json
    title = data['title']
    content = data['content']
    date = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (title, content, link, date) VALUES (?, ?, ?, ?)",
              (title, content, None, date))
    doc_id = c.lastrowid
    conn.commit()
    store_embedding(str(doc_id), content)
    conn.close()
    return {"message": "Note added"}

@app.route('/add_task', methods=['POST'])
def add_task():
    data = request.json
    title = data['title']
    description = data['description']
    status = data['status']
    date = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("INSERT INTO tasks (title, description, status, date) VALUES (?, ?, ?, ?)",
              (title, description, status, date))
    task_id = c.lastrowid
    conn.commit()
    store_embedding(f"task_{task_id}", f"{title} {description}")
    conn.close()
    return {"message": "Task added"}

@app.route('/get_tasks', methods=['GET'])
def get_tasks():
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("SELECT id, title, description, status, date FROM tasks")
    tasks = [{"id": row[0], "title": row[1], "description": row[2], "status": row[3], "date": row[4]} for row in c.fetchall()]
    conn.close()
    return {"tasks": tasks}

@app.route('/delete_task', methods=['POST'])
def delete_task():
    data = request.json
    task_id = data['id']
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Delete the task
    c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    
    # Delete associated embedding
    c.execute("DELETE FROM embeddings WHERE doc_id = ?", (f"task_{task_id}",))
    
    conn.commit()
    conn.close()
    
    return {"message": "Task deleted successfully"}

@app.route('/get_task', methods=['GET'])
def get_task():
    task_id = request.args.get('id', type=int)
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    c.execute("SELECT id, title, description, status, date FROM tasks WHERE id = ?", (task_id,))
    result = c.fetchone()
    
    conn.close()
    
    if result:
        return {
            "id": result[0],
            "title": result[1],
            "description": result[2],
            "status": result[3],
            "date": result[4]
        }
    else:
        return {"error": "Task not found"}, 404

@app.route('/update_task', methods=['POST'])
def update_task():
    data = request.json
    task_id = data['id']
    title = data['title']
    description = data['description']
    status = data['status']
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Update the task
    c.execute("UPDATE tasks SET title = ?, description = ?, status = ? WHERE id = ?", 
              (title, description, status, task_id))
    
    # Update embedding
    # Delete old embedding
    c.execute("DELETE FROM embeddings WHERE doc_id = ?", (f"task_{task_id}",))
    conn.commit()
    
    # Create new embedding
    store_embedding(f"task_{task_id}", f"{title} {description}")
    
    conn.commit()
    conn.close()
    
    return {"message": "Task updated successfully"}

@app.route('/update_task_status', methods=['POST'])
def update_task_status():
    data = request.json
    task_id = data['id']
    status = data['status']
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Update the task status
    c.execute("UPDATE tasks SET status = ? WHERE id = ?", 
              (status, task_id))
    
    conn.commit()
    conn.close()
    
    return {"message": "Task status updated successfully"}

@app.route('/get_documents', methods=['GET'])
def get_documents():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)
    offset = (page - 1) * per_page
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Get total count
    c.execute("SELECT COUNT(*) FROM documents WHERE link IS NOT NULL")
    total_count = c.fetchone()[0]
    
    # Get paginated documents with links (external documents)
    c.execute("SELECT id, title, link, date FROM documents WHERE link IS NOT NULL ORDER BY id DESC LIMIT ? OFFSET ?", 
              (per_page, offset))
    documents = [{"id": row[0], "title": row[1], "link": row[2], "date": row[3]} for row in c.fetchall()]
    
    conn.close()
    
    total_pages = (total_count + per_page - 1) // per_page  # Ceiling division
    
    return {
        "documents": documents,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

@app.route('/get_notes', methods=['GET'])
def get_notes():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)
    offset = (page - 1) * per_page
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Get total count
    c.execute("SELECT COUNT(*) FROM documents WHERE link IS NULL")
    total_count = c.fetchone()[0]
    
    # Get paginated notes (documents without links)
    c.execute("SELECT id, title, content, date FROM documents WHERE link IS NULL ORDER BY id DESC LIMIT ? OFFSET ?", 
              (per_page, offset))
    notes = [{"id": row[0], "title": row[1], "content": row[2], "date": row[3]} for row in c.fetchall()]
    
    conn.close()
    
    total_pages = (total_count + per_page - 1) // per_page  # Ceiling division
    
    return {
        "notes": notes,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

@app.route('/delete_document', methods=['POST'])
def delete_document():
    data = request.json
    doc_id = data['id']
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Delete the document
    c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    
    # Delete associated embedding
    c.execute("DELETE FROM embeddings WHERE doc_id = ?", (str(doc_id),))
    
    conn.commit()
    conn.close()
    
    return {"message": "Document deleted successfully"}

@app.route('/get_document', methods=['GET'])
def get_document():
    doc_id = request.args.get('id', type=int)
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Check if it's a document with a link
    c.execute("SELECT id, title, link, content, date FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    
    conn.close()
    
    if result:
        return {
            "id": result[0],
            "title": result[1],
            "link": result[2],
            "content": result[3],
            "date": result[4]
        }
    else:
        return {"error": "Document not found"}, 404

@app.route('/update_document', methods=['POST'])
def update_document():
    data = request.json
    doc_id = data['id']
    title = data['title']
    content = data.get('content', '')
    link = data.get('link', None)
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Update the document
    c.execute("UPDATE documents SET title = ?, content = ?, link = ? WHERE id = ?", 
              (title, content, link, doc_id))
    
    # Update embedding if content changed substantially
    if content:
        # Delete old embedding
        c.execute("DELETE FROM embeddings WHERE doc_id = ?", (str(doc_id),))
        conn.commit()
        
        # Create new embedding
        store_embedding(str(doc_id), content)
    
    conn.commit()
    conn.close()
    
    return {"message": "Document updated successfully"}

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']
    query_embedding = model.encode(query)
    
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    c.execute("SELECT doc_id, embedding FROM embeddings")
    embeddings = c.fetchall()
    
    results = []
    for doc_id, emb_bytes in embeddings:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        if doc_id.startswith("task_"):
            task_id = int(doc_id.split("_")[1])
            c.execute("SELECT title, description, date FROM tasks WHERE id = ?", (task_id,))
            title, description, date = c.fetchone()
            results.append({"title": title, "description": description, "date": date, "similarity": float(similarity), "type": "task"})
        else:
            c.execute("SELECT title, link, date FROM documents WHERE id = ?", (int(doc_id),))
            title, link, date = c.fetchone()
            results.append({"id": int(doc_id), "title": title, "link": link, "date": date, "similarity": float(similarity), "type": "document"})
    
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:5]
    conn.close()
    return {"results": results}

if __name__ == '__main__':
    init_db()
    app.run(debug=True)