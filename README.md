# AI Counseling Chatbot (Rasa)

This chatbot helps answer queries about college admission, courses, and FAQs using Rasa and PDF-based semantic search.

## Features
- NLP-powered Q&A
- Rasa custom actions
- PDF chunk search with FAISS
- Flask frontend

## Setup

```bash
pip install -r requirements.txt
rasa train
rasa run actions &
rasa shell
