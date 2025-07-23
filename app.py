# UIET Chatbot Flask App with PDF + TXT support, Rasa, FAQ, and BERT QA

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import fitz  # PyMuPDF
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
import uuid
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production
CORS(app)
logging.basicConfig(level=logging.INFO, filename="chatbot.log")

# Configuration
PDF_FOLDER = "pdfs"
TEXT_FOLDER = "Texts"
INDEX_PATH = "pdf_index.faiss"
CHUNKS_PATH = "pdf_chunks.pkl"
RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005/webhooks/rest/webhook")

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cache = TTLCache(maxsize=100, ttl=3600)

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1
bert_qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=device)

index = None
chunks = []
chunk_embeddings = None  # Cache for chunk embeddings to avoid repeated computation
pdf_response_state = {"chunks": [], "pointer": 0}

# Load FAQ
with open("college_faq.json", "r", encoding="utf-8") as f:
    saved_faq = json.load(f)

# Load previous posts
POSTS_FILE = "forum_posts.json"
if os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)
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

def process_all_documents():
    global index, chunks, chunk_embeddings
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print(f"✅ Using cached FAISS index from '{INDEX_PATH}'")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        # Compute and cache chunk embeddings once after loading chunks
        chunk_embeddings = EMBEDDING_MODEL.encode(chunks, show_progress_bar=True)
        print(f"✅ Loaded {len(chunks)} chunks and cached embeddings from cache")
        return

    print("🔄 Extracting chunks...")
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
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        chunks = all_chunks
        chunk_embeddings = embeddings  # Cache embeddings
        print(f"✅ Indexed {len(chunks)} chunks and cached embeddings.")
    else:
        print("⚠️ No chunks to index.")

def get_pdf_response(message, k=5):
    if not index or not chunks or chunk_embeddings is None:
        return None
    try:
        query_embedding = EMBEDDING_MODEL.encode([message])
        distances, indices = index.search(query_embedding, k * 10)
        filtered_chunks, filtered_embeddings = [], []
        for idx in indices[0]:
            if idx >= len(chunks):
                continue
            chunk = chunks[idx]
            emb = chunk_embeddings[idx]
            sim_scores = [np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e)) for e in filtered_embeddings]
            if not any(score > 0.9 for score in sim_scores):
                filtered_chunks.append(chunk)
                filtered_embeddings.append(emb)
            if len(filtered_chunks) >= k:
                break
        pdf_response_state["chunks"] = filtered_chunks
        pdf_response_state["pointer"] = min(2, len(filtered_chunks))
        if len(filtered_chunks) == 0:
            return None
        return filtered_chunks[:2]
    except Exception as e:
        logging.error(f"PDF search error: {e}")
        return None

def generate_bert_answer(question, context_chunks):
    best_answer = ""
    best_score = 0
    for chunk in context_chunks:
        result = bert_qa(question=question, context=chunk)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]
    return best_answer if best_score > 0.3 else None

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

def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message}, timeout=5)
        response.raise_for_status()
        return [msg.get("text", "") for msg in response.json() if msg.get("text")]
    except Exception as e:
        logging.error(f"Rasa error: {e}")
        return []

@app.route("/")
def landing():
    return render_template("landing.html")

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

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    if user_message in ["continue", "more"]:
        start = pdf_response_state["pointer"]
        remaining = pdf_response_state["chunks"][start:]
        if remaining:
            next_chunks = remaining[:2]
            pdf_response_state["pointer"] += len(next_chunks)
            return jsonify({"response": next_chunks})
        else:
            return jsonify({"response": ["No more information to show."]})

    # Step 1: Try FAQ
    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})

    # Step 2: Try PDF + DistilBERT answer
    pdf_chunks = get_pdf_response(user_message)
    if pdf_chunks:
        bert_answer = generate_bert_answer(user_message, pdf_chunks)
        if bert_answer:
            return jsonify({"response": [bert_answer]})

    # Step 3: Try Rasa
    rasa_responses = get_rasa_response(user_message)
    if rasa_responses:
        return jsonify({"response": rasa_responses})

    # Step 4: Fallback
    return jsonify({"response": ["Sorry, I couldn’t find any relevant information, You can visit https://uiet.puchd.ac.in/ for more details or ask your doubt in forum section."]})



@app.route("/post_question", methods=["POST"])
@login_required
def post_question():
    question_text = request.form.get("question_text", "").strip()
    if not question_text:
        return redirect(url_for("forum"))
    question_id = str(uuid.uuid4())
    user = session.get("username")
    user_type = users.get(user, {}).get("type", "Student")
    posts.append({
        "id": question_id,
        "question": question_text,
        "user": user,
        "user_type": user_type,
        "answers": []
    })

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
            user_type = users.get(username, {}).get("type", "Student")
            post["answers"].append({
            "id": answer_id,
            "text": answer_text,
            "user": username,
            "user_type": user_type,
            "verified": False,
            "votes": 0
})

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

if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    process_all_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)
