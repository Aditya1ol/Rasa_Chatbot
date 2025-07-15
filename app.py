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


import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure key in production
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

# In-memory user store (for demo purposes)
users = {
    "adi_admin": {
        "password": "Myfirstchatbot6999",
        "email": "anayyer50@gmail.com",
        "is_admin": True
    },
    "student1": {"password": "studentpass", "is_admin": False}
}

# In-memory posts store
posts = []

# Utility: Normalize text spacing

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

# Admin check decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or not users.get(session['username'], {}).get('is_admin', False):
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

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

# Forum page
@app.route("/forum", methods=["GET"])
def forum():
    logged_in = 'username' in session
    is_admin = False
    if logged_in:
        is_admin = users.get(session['username'], {}).get('is_admin', False)
    return render_template("forum.html", posts=posts, logged_in=logged_in, is_admin=is_admin)

# Authentication (login/signup) handler
@app.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "GET":
        return render_template("login.html")
    error = None
    user_type = request.form.get("user_type")
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")
    branch = request.form.get("branch")
    year = request.form.get("year")

    if not user_type or not username or not email or not password:
        error = "Please fill in all required fields."
        return render_template("login.html", error=error)

    # Admin credentials hardcoded
    if user_type == "admin":
        admin_user = users.get("adi_admin")
        if username == "adi_admin" and password == admin_user["password"]:
            session['username'] = "adi_admin"
            return redirect(url_for('forum'))
        else:
            error = "Invalid admin username or password."
            return render_template("login.html", error=error)

    # For college students and query users, handle signup/login
    user = users.get(username)
    if user:
        # User exists, check password and type
        if user["password"] == password and user.get("type") == user_type:
            session['username'] = username
            return redirect(url_for('forum'))
        else:
            error = "Invalid username, password, or user type."
            return render_template("login.html", error=error)
    else:
        # New user signup
        new_user = {
            "password": password,
            "type": user_type,
            "email": email
        }
        if user_type == "college_student":
            new_user["branch"] = branch
            new_user["year"] = year
        users[username] = new_user
        session['username'] = username
        return redirect(url_for('forum'))

    return render_template("login.html", error=error)

# Login page rendering
@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

# Logout
@app.route("/logout")
@login_required
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

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

# Post an answer to a forum question
@app.route("/post_answer", methods=["POST"])
@login_required
def post_answer():
    post_id = request.form.get("post_id")
    answer_text = request.form.get("answer_text", "").strip()
    username = session.get("username")

    if not post_id or not answer_text:
        return redirect(url_for('forum'))

    user = users.get(username, {})
    if user.get("type") != "college_student" and not user.get("is_admin", False):
        # Only college students and admins can answer questions
        return redirect(url_for('forum'))

    # Find the post
    for post in posts:
        if post["id"] == post_id:
            answer_id = str(uuid.uuid4())
            answer_entry = {
                "id": answer_id,
                "text": answer_text,
                "user": username,
                "verified": False,
                "votes": 0
            }
            post["answers"].append(answer_entry)
            break
    else:
        # Post not found
        return redirect(url_for('forum'))

    return redirect(url_for('forum'))

# Post a new question to the forum
@app.route("/post_question", methods=["POST"])
@login_required
def post_question():
    question_text = request.form.get("question_text", "").strip()
    username = session.get("username")

    if not question_text:
        return redirect(url_for('forum'))

    user = users.get(username, {})
    if user.get("type") == "query_user" or user.get("is_admin", False):
        # Only query users and admins can post new questions
        question_id = str(uuid.uuid4())
        new_post = {
            "id": question_id,
            "question": question_text,
            "answers": []
        }
        posts.append(new_post)
        return redirect(url_for('forum'))
    else:
        # Not permitted to post questions
        return redirect(url_for('forum'))

# Verify an answer (admin only)
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
                    break
            break

    return redirect(url_for('forum'))

# Vote for an answer (admin only)
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
                    break
            break

    return redirect(url_for('forum'))

# Delete a post (admin only)
@app.route("/delete_post", methods=["POST"])
@admin_required
def delete_post():
    post_id = request.form.get("post_id")
    global posts
    posts = [post for post in posts if post["id"] != post_id]
    return redirect(url_for('forum'))

# Delete an answer (admin only)
@app.route("/delete_answer", methods=["POST"])
@admin_required
def delete_answer():
    post_id = request.form.get("post_id")
    answer_id = request.form.get("answer_id")

    for post in posts:
        if post["id"] == post_id:
            post["answers"] = [ans for ans in post["answers"] if ans["id"] != answer_id]
            break

    return redirect(url_for('forum'))

# Submit contact info of current students
@app.route("/submit_contact_info", methods=["POST"])
def submit_contact_info():
    data = request.json
    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    phone = data.get("phone", "").strip()
    department = data.get("department", "").strip()

    if not name or not email:
        return jsonify({"success": False, "message": "Name and email are required."})

    contact_entry = {
        "name": name,
        "email": email,
        "phone": phone,
        "department": department
    }

    contacts_file = "students_contacts.json"
    contacts = []
    if os.path.exists(contacts_file):
        with open(contacts_file, "r", encoding="utf-8") as f:
            try:
                contacts = json.load(f)
            except:
                contacts = []

    contacts.append(contact_entry)
    with open(contacts_file, "w", encoding="utf-8") as f:
        json.dump(contacts, f, indent=2)

    return jsonify({"success": True, "message": "Contact information submitted successfully."})

# Student chat endpoint (simulate sending or storing messages)
@app.route("/student_chat", methods=["POST"])
def student_chat():
    user_message = request.json.get("message", "").strip()
    # For now, just echo back the message with a placeholder response
    response_messages = [
        "Thank you for your question! A current student will get back to you soon.",
        "Meanwhile, feel free to explore the chatbot or contact us page."
    ]
    return jsonify({"response": response_messages})

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
