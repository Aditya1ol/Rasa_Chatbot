# AI Counseling Chatbot (Rasa)

This chatbot helps answer queries about UIET's - college admission, courses, and FAQs using Rasa for intent based queries and Google's BERT + FAISS for PDF-based semantic search.

## Features
- NLP-powered Q&A
- Rasa custom actions
- PDF chunk search with FAISS
- BERT for conversational flow
- Flask API
  

## Setup

```bash
pip install -r requirements.txt
rasa train
rasa run actions
rasa run --cors "*" --enable-api
python app.py

