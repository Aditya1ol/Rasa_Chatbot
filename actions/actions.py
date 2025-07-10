from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

class ActionFetchPDFInfo(Action):

    def name(self):
        return "action_fetch_pdf_info"

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("pdf_index.faiss")
        with open("pdf_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message.get("text")
        embedding = self.model.encode([query])
        _, I = self.index.search(np.array(embedding), k=1)
        chunk = self.chunks[I[0][0]]

        # Extract only the answer part from the chunk
        answer_match = re.search(r'Answer:\s*(.*?)(?=Question\s*\d+:|$)', chunk, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = chunk.strip()

        dispatcher.utter_message(text=answer)
        return []

class ActionGenerateT5Response(Action):

    def name(self):
        return "action_generate_t5_response"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "actions/trained_model"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        inputs = self.tokenizer.encode(user_message, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model.generate(inputs, max_length=100, num_beams=5, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        dispatcher.utter_message(text=response)
        return []
