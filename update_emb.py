import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

def migrate_embeddings():
    print("Starting migration of embeddings...")
    
    # Load the sentence transformer model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to the database
    conn = sqlite3.connect('docs.db')
    c = conn.cursor()
    
    # Get all documents
    print("Fetching documents...")
    c.execute("SELECT id, title, content, link FROM documents")
    documents = c.fetchall()
    
    # Get all tasks
    print("Fetching tasks...")
    c.execute("SELECT id, title, description FROM tasks")
    tasks = c.fetchall()
    
    # Create a function to store embeddings
    def store_embedding(doc_id, content):
        print(f"Creating embedding for {doc_id}...")
        embedding = model.encode(content)
        embedding_bytes = embedding.tobytes()
        
        # Check if embedding exists
        c.execute("SELECT COUNT(*) FROM embeddings WHERE doc_id = ?", (doc_id,))
        exists = c.fetchone()[0] > 0
        
        if exists:
            c.execute("UPDATE embeddings SET embedding = ? WHERE doc_id = ?", 
                     (embedding_bytes, doc_id))
            print(f"Updated embedding for {doc_id}")
        else:
            c.execute("INSERT INTO embeddings (doc_id, embedding) VALUES (?, ?)", 
                     (doc_id, embedding_bytes))
            print(f"Created new embedding for {doc_id}")
    
    # Process documents
    print(f"Processing {len(documents)} documents...")
    for doc_id, title, content, link in documents:
        if link is not None:
            # For linked documents, use the title
            store_embedding(str(doc_id), title)
        else:
            # For regular documents and notes, use title + content
            store_embedding(str(doc_id), f"{title} {content}")
    
    # Process tasks
    print(f"Processing {len(tasks)} tasks...")
    for task_id, title, description in tasks:
        store_embedding(f"task_{task_id}", f"{title} {description}")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Migration completed successfully!")
    print(f"Updated {len(documents)} documents and {len(tasks)} tasks")

if __name__ == "__main__":
    migrate_embeddings()