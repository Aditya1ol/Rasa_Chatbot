# utils.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from flask import session

# For stateful 'continue' command
last_indices = {}

def load_index():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss_index.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model

def embed_query(query, model):
    return model.encode([query])

def search_faiss(query, model, index, chunks, session_id, top_k=3):
    global last_indices
    if query == "continue" and session_id in last_indices:
        last_shown = last_indices[session_id]
        next_indices = last_shown + 1
        if next_indices < len(chunks):
            last_indices[session_id] = next_indices
            return chunks[next_indices]
        else:
            return "No more content to continue."
    else:
        query_vector = embed_query(query, model)
        distances, indices = index.search(query_vector, top_k)
        result_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        if result_chunks:
            last_indices[session_id] = indices[0][0]
            return " ".join(result_chunks)
        else:
            return "I'm sorry, I couldn't find a relevant answer in the documents."

def get_rasa_response(message, sender_id):
    url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {
        "sender": sender_id,
        "message": message
    }
    try:
        response = requests.post(url, json=payload)
        bot_messages = response.json()
        if bot_messages:
            return bot_messages[0].get("text", "Sorry, I didn't understand that.")
        else:
            return "Sorry, no response from Rasa."
    except:
        return "Rasa server is not reachable."
