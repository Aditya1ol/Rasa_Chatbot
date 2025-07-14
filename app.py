# UIET Chatbot Flask App with PDF + TXT support, Rasa, FAQ, and BERT QA

from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for reading PDFs
import re
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
import requests
import logging
from cachetools import TTLCache
from transformers import pipeline
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, filename="chatbot.log")

# Configuration
PDF_FOLDER = "pdfs"              # PDF document folder
TEXT_FOLDER = "Texts"            # Text document folder
INDEX_PATH = "pdf_index.faiss"   # FAISS vector index path
CHUNKS_PATH = "pdf_chunks.pkl"   # Stored chunks for later use
RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005/webhooks/rest/webhook")

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cache = TTLCache(maxsize=100, ttl=3600)

# Use GPU if available for BERT QA
device = 0 if torch.cuda.is_available() else -1
bert_qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device=device)

# Global variables for FAISS
index = None
chunks = []

# Track PDF response state for chunk continuation
pdf_response_state = {"chunks": [], "pointer": 0}

# Load FAQ JSON
with open("college_faq.json", "r", encoding="utf-8") as f:
    saved_faq = json.load(f)

# Load previously generated questions
GENERATED_QUESTIONS_PATH = "generated_questions.json"
if os.path.exists(GENERATED_QUESTIONS_PATH):
    with open(GENERATED_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        generated_questions = json.load(f)
else:
    generated_questions = []

# Utility: Normalize text spacing

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Extract chunks from PDF pages
def extract_chunks_from_pdf(pdf_path, window_size=3, stride=1):
    chunks = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        text = clean_text(text)
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = " ".join(sentences[i:i + window_size])
            if len(chunk) > 30:
                chunks.append(chunk.strip())
    return chunks

# Extract chunks from plain text files
def extract_chunks_from_txt(txt_path, window_size=3, stride=1):
    chunks = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = clean_text(text)
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = " ".join(sentences[i:i + window_size])
            if len(chunk) > 30:
                chunks.append(chunk.strip())
    except Exception as e:
        logging.error(f"Failed to process {txt_path}: {e}")
    return chunks

# Process and index all PDFs and TXT files into FAISS
def process_all_documents():
    global index, chunks

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print(f"✅ Using cached FAISS index from '{INDEX_PATH}'")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks from cache")
        return

    print("🔄 No cache found. Extracting chunks from documents...")
    all_chunks = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            all_chunks.extend(extract_chunks_from_pdf(path))

    for filename in os.listdir(TEXT_FOLDER):
        if filename.lower().endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, filename)
            all_chunks.extend(extract_chunks_from_txt(path))

    if all_chunks:
        print(f"✅ Extracted {len(all_chunks)} total chunks from documents.")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        chunks = all_chunks
    else:
        print("⚠️ No content found to process.")

# Search relevant chunks for a user message
def get_pdf_response(message, k=5):
    if not index or not chunks:
        return ["No PDF data found."]
    try:
        query_embedding = EMBEDDING_MODEL.encode([message])
        distances, indices = index.search(query_embedding, k * 10)

        candidate_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]

        filtered_chunks = []
        filtered_embeddings = []
        for chunk in candidate_chunks:
            chunk_embedding = EMBEDDING_MODEL.encode([chunk])[0]
            similarity_scores = [np.dot(chunk_embedding, fe) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(fe)) for fe in filtered_embeddings]
            if not any(score > 0.9 for score in similarity_scores):
                filtered_chunks.append(chunk)
                filtered_embeddings.append(chunk_embedding)
            if len(filtered_chunks) >= k:
                break

        pdf_response_state["chunks"] = filtered_chunks
        pdf_response_state["pointer"] = min(2, len(filtered_chunks))
        return filtered_chunks[:2]
    except Exception as e:
        logging.error(f"PDF search error: {e}")
        return ["Error processing PDF."]

# Use BERT QA to generate best answer from chunks
def generate_bert_answer(question, context_chunks):
    best_answer = ""
    best_score = 0
    for chunk in context_chunks:
        result = bert_qa(question=question, context=chunk)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]
    return best_answer if best_score > 0.3 else None

# Semantic FAQ answer matching using embedding similarity
def find_faq_answer(user_message, threshold=0.75):
    if not saved_faq:
        return None
    questions = [faq["question"] for faq in saved_faq]
    question_embeddings = EMBEDDING_MODEL.encode(questions)
    query_embedding = EMBEDDING_MODEL.encode(user_message)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    return saved_faq[best_idx]["answer"] if best_score >= threshold else None

# Call Rasa to get smalltalk response
def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message}, timeout=5)
        response.raise_for_status()
        return [msg.get("text", "") for msg in response.json() if msg.get("text")]
    except Exception as e:
        logging.error(f"Rasa error: {e}")
        return []

# Landing page
@app.route("/")
def landing():
    return render_template("landing.html")

# Main chatbot interface
@app.route("/chatbot")
def home():
    return render_template("index.html", saved_questions=saved_faq, generated_questions=generated_questions)

# Main chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    if user_message in ["continue", "more"]:
        start = pdf_response_state["pointer"]
        remaining_chunks = pdf_response_state["chunks"][start:]
        if remaining_chunks:
            next_chunks = remaining_chunks[:2]
            pdf_response_state["pointer"] += len(next_chunks)
            return jsonify({"response": next_chunks})
        else:
            return jsonify({"response": ["No more information to show."]})
        


    # Priority 1: FAQ match
    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})
    
    
    # Priority 3: BERT + semantic chunk matching
    pdf_chunks = get_pdf_response(user_message)
    if pdf_chunks and "No PDF data found." not in pdf_chunks[0]:
        bert_answer = generate_bert_answer(user_message, pdf_chunks)
        if bert_answer:
            return jsonify({"response": [bert_answer]})    

    # Priority 2: Rasa fallback
    rasa_responses = get_rasa_response(user_message)
    if rasa_responses:
        return jsonify({"response": rasa_responses})



    # Final fallback
    return jsonify({"response": ["Sorry, I couldn’t find any relevant information."]})

# Save a new FAQ item
@app.route("/save_faq", methods=["POST"])
def save_faq():
    new_faq = request.json.get("faq", {})
    saved_faq.append(new_faq)
    with open("college_faq.json", "w", encoding="utf-8") as f:
        json.dump(saved_faq, f, indent=2)
    return jsonify({"message": "FAQ saved successfully"})

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    feedback_text = data.get("feedback", "").strip()

    if feedback_text:
        # Save feedback to a file
        with open("user_feedback.json", "a", encoding="utf-8") as f:
            json.dump({"feedback": feedback_text}, f)
            f.write("\n")  # Newline for readability
        return jsonify({"message": "✅ Feedback submitted. Thank you!"})
    else:
        return jsonify({"message": "⚠️ Feedback was empty."})


# Main entry point
if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"📁 Created folder: {PDF_FOLDER}")
    if not os.path.exists(TEXT_FOLDER):
        os.makedirs(TEXT_FOLDER)
        print(f"📁 Created folder: {TEXT_FOLDER}")
    process_all_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)
