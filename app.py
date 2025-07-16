# UIET Chatbot Flask App with PDF + TXT support, Rasa, FAQ, and BERT QA

<<<<<<< HEAD
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import fitz  # PyMuPDF
=======
from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF for reading PDFs
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
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
<<<<<<< HEAD
import uuid
from functools import wraps
=======
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production
CORS(app)
logging.basicConfig(level=logging.INFO, filename="chatbot.log")

# Configuration
<<<<<<< HEAD
PDF_FOLDER = "pdfs"
TEXT_FOLDER = "Texts"
INDEX_PATH = "pdf_index.faiss"
CHUNKS_PATH = "pdf_chunks.pkl"
RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005/webhooks/rest/webhook")

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cache = TTLCache(maxsize=100, ttl=3600)

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1
bert_qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device=device)

index = None
chunks = []
pdf_response_state = {"chunks": [], "pointer": 0}

# Load FAQ
with open("college_faq.json", "r", encoding="utf-8") as f:
    saved_faq = json.load(f)

# Load previous posts
POSTS_FILE = "forum_posts.json"
if os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)
=======
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
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
else:
    posts = []

# Load users (demo)
users = {
    "adi_admin": {
        "password": "Myfirstchatbot6999",
        "email": "anayyer50@gmail.com",
        "is_admin": True
    },
    "student1": {"password": "studentpass", "is_admin": False}
}

# Utility

<<<<<<< HEAD
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or not users.get(session['username'], {}).get('is_admin', False):
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

=======
# Utility: Normalize text spacing

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Extract chunks from PDF pages
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
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

<<<<<<< HEAD
=======
# Extract chunks from plain text files
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
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

<<<<<<< HEAD
def process_all_documents():
    global index, chunks
=======
# Process and index all PDFs and TXT files into FAISS
def process_all_documents():
    global index, chunks

>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print(f"✅ Using cached FAISS index from '{INDEX_PATH}'")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks from cache")
        return

<<<<<<< HEAD
    print("🔄 Extracting chunks...")
=======
    print("🔄 No cache found. Extracting chunks from documents...")
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    all_chunks = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            all_chunks.extend(extract_chunks_from_pdf(path))
<<<<<<< HEAD
=======

>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    for filename in os.listdir(TEXT_FOLDER):
        if filename.lower().endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, filename)
            all_chunks.extend(extract_chunks_from_txt(path))

    if all_chunks:
<<<<<<< HEAD
=======
        print(f"✅ Extracted {len(all_chunks)} total chunks from documents.")
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        chunks = all_chunks
<<<<<<< HEAD
        print(f"✅ Indexed {len(chunks)} chunks.")
    else:
        print("⚠️ No chunks to index.")

=======
    else:
        print("⚠️ No content found to process.")

# Search relevant chunks for a user message
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
def get_pdf_response(message, k=5):
    if not index or not chunks:
        return ["No PDF data found."]
    try:
        query_embedding = EMBEDDING_MODEL.encode([message])
        distances, indices = index.search(query_embedding, k * 10)
<<<<<<< HEAD
        candidate_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
        filtered_chunks, filtered_embeddings = [], []
        for chunk in candidate_chunks:
            emb = EMBEDDING_MODEL.encode([chunk])[0]
            sim_scores = [np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e)) for e in filtered_embeddings]
            if not any(score > 0.9 for score in sim_scores):
=======

        candidate_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]

        filtered_chunks = []
        filtered_embeddings = []
        for chunk in candidate_chunks:
            chunk_embedding = EMBEDDING_MODEL.encode([chunk])[0]
            similarity_scores = [np.dot(chunk_embedding, fe) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(fe)) for fe in filtered_embeddings]
            if not any(score > 0.9 for score in similarity_scores):
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
                filtered_chunks.append(chunk)
                filtered_embeddings.append(emb)
            if len(filtered_chunks) >= k:
                break
        pdf_response_state["chunks"] = filtered_chunks
        pdf_response_state["pointer"] = min(2, len(filtered_chunks))
        return filtered_chunks[:2]
    except Exception as e:
        logging.error(f"PDF search error: {e}")
        return ["Error processing PDF."]

<<<<<<< HEAD
=======
# Use BERT QA to generate best answer from chunks
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
def generate_bert_answer(question, context_chunks):
    best_answer = ""
    best_score = 0
    for chunk in context_chunks:
        result = bert_qa(question=question, context=chunk)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]
    return best_answer if best_score > 0.3 else None

<<<<<<< HEAD
=======
# Semantic FAQ answer matching using embedding similarity
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
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
<<<<<<< HEAD

def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message}, timeout=5)
        response.raise_for_status()
        return [msg.get("text", "") for msg in response.json() if msg.get("text")]
    except Exception as e:
        logging.error(f"Rasa error: {e}")
        return []
=======
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8

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
    return render_template("index.html", saved_questions=saved_faq)

@app.route("/forum")
def forum():
    logged_in = 'username' in session
    is_admin = users.get(session.get('username'), {}).get("is_admin", False)
    return render_template("forum.html", posts=posts, logged_in=logged_in, is_admin=is_admin)

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "GET":
        return render_template("login.html")
    username = request.form.get("username")
    password = request.form.get("password")
    user_type = request.form.get("user_type")
    branch = request.form.get("branch")
    year = request.form.get("year")
    email = request.form.get("email")

    if username in users:
        user = users[username]
        if user["password"] == password:
            session['username'] = username
            return redirect(url_for("forum"))
        else:
            return render_template("login.html", error="Invalid password.")
    else:
        users[username] = {"password": password, "type": user_type, "email": email, "branch": branch, "year": year, "is_admin": False}
        session['username'] = username
        return redirect(url_for("forum"))

@app.route("/logout")
@login_required
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

# Main chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    if user_message in ["continue", "more"]:
        start = pdf_response_state["pointer"]
<<<<<<< HEAD
        remaining = pdf_response_state["chunks"][start:]
        if remaining:
            next_chunks = remaining[:2]
=======
        remaining_chunks = pdf_response_state["chunks"][start:]
        if remaining_chunks:
            next_chunks = remaining_chunks[:2]
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
            pdf_response_state["pointer"] += len(next_chunks)
            return jsonify({"response": next_chunks})
        else:
            return jsonify({"response": ["No more information to show."]})
<<<<<<< HEAD

    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})

=======
        


    # Priority 1: FAQ match
    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})
    
    
    # Priority 3: BERT + semantic chunk matching
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    pdf_chunks = get_pdf_response(user_message)
    if pdf_chunks and "No PDF data found." not in pdf_chunks[0]:
        bert_answer = generate_bert_answer(user_message, pdf_chunks)
        if bert_answer:
<<<<<<< HEAD
            return jsonify({"response": [bert_answer]})

=======
            return jsonify({"response": [bert_answer]})    

    # Priority 2: Rasa fallback
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    rasa_responses = get_rasa_response(user_message)
    if rasa_responses:
        return jsonify({"response": rasa_responses})

<<<<<<< HEAD
    return jsonify({"response": ["Sorry, I couldn’t find any relevant information."]})

@app.route("/post_question", methods=["POST"])
@login_required
def post_question():
    question_text = request.form.get("question_text", "").strip()
    if not question_text:
        return redirect(url_for("forum"))
    question_id = str(uuid.uuid4())
    posts.append({"id": question_id, "question": question_text, "answers": []})
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/post_answer", methods=["POST"])
@login_required
def post_answer():
    post_id = request.form.get("post_id")
    answer_text = request.form.get("answer_text", "").strip()
    username = session.get("username")
    if not post_id or not answer_text:
        return redirect(url_for("forum"))
    for post in posts:
        if post["id"] == post_id:
            answer_id = str(uuid.uuid4())
            post["answers"].append({"id": answer_id, "text": answer_text, "user": username, "verified": False, "votes": 0})
            break
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/verify_answer", methods=["POST"])
@admin_required
def verify_answer():
    post_id = request.form.get("post_id")
    answer_id = request.form.get("answer_id")
    for post in posts:
        if post["id"] == post_id:
            for answer in post["answers"]:
                if answer["id"] == answer_id:
                    answer["verified"] = True
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/vote_answer", methods=["POST"])
@admin_required
def vote_answer():
    post_id = request.form.get("post_id")
    answer_id = request.form.get("answer_id")
    for post in posts:
        if post["id"] == post_id:
            for answer in post["answers"]:
                if answer["id"] == answer_id:
                    answer["votes"] += 1
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/delete_post", methods=["POST"])
@admin_required
def delete_post():
    post_id = request.form.get("post_id")
    global posts
    posts = [p for p in posts if p["id"] != post_id]
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/delete_answer", methods=["POST"])
@admin_required
def delete_answer():
    post_id = request.form.get("post_id")
    answer_id = request.form.get("answer_id")
    for post in posts:
        if post["id"] == post_id:
            post["answers"] = [a for a in post["answers"] if a["id"] != answer_id]
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2)
    return redirect(url_for("forum"))

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    feedback_text = data.get("feedback", "").strip()
    if feedback_text:
        with open("user_feedback.json", "a", encoding="utf-8") as f:
            json.dump({"feedback": feedback_text}, f)
            f.write("\n")
        return jsonify({"message": "✅ Feedback submitted. Thank you!"})
    else:
        return jsonify({"message": "⚠️ Feedback was empty."})
=======


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
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8

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
<<<<<<< HEAD
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)
=======
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"📁 Created folder: {PDF_FOLDER}")
    if not os.path.exists(TEXT_FOLDER):
        os.makedirs(TEXT_FOLDER)
        print(f"📁 Created folder: {TEXT_FOLDER}")
>>>>>>> 8bf9dbbcafd16d47b7bbacf9b4940e977436e1b8
    process_all_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)
