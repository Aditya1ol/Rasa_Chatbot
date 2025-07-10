import os
import json
from extract_pdf import extract_chunks_from_pdf
from transformers import pipeline

PDF_FOLDER = "pdfs"
FAQ_JSON_PATH = "college_faq.json"

def load_existing_faq():
    if os.path.exists(FAQ_JSON_PATH):
        with open(FAQ_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_faq(faq_list):
    with open(FAQ_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(faq_list, f, indent=2, ensure_ascii=False)

def generate_qa_pairs(chunks):
    from transformers import pipeline
    qa_pairs = []
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-e2e-qg")
    for chunk in chunks:
        try:
            # The model expects input text and generates questions and answers in text format
            outputs = question_generator(chunk, max_length=256, truncation=True)
            for output in outputs:
                text = output.get("generated_text", "")
                # The output text format is "question: ... answer: ..."
                if "question:" in text and "answer:" in text:
                    q_start = text.find("question:") + len("question:")
                    a_start = text.find("answer:")
                    question = text[q_start:a_start].strip()
                    answer = text[a_start + len("answer:"):].strip()
                    if question and answer:
                        qa_pairs.append({"question": question, "answer": answer})
        except Exception as e:
            print(f"Error generating Q&A for chunk: {e}")
    return qa_pairs

def merge_faqs(existing, new):
    existing_questions = set(item['question'] for item in existing)
    merged = existing[:]
    for item in new:
        if item['question'] not in existing_questions:
            merged.append(item)
    return merged

def main():
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing {filename}...")
            try:
                chunks = extract_chunks_from_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"Extracted {len(all_chunks)} chunks from PDFs.")

    new_qa_pairs = generate_qa_pairs(all_chunks)
    print(f"Generated {len(new_qa_pairs)} new Q&A pairs.")

    existing_faq = load_existing_faq()
    merged_faq = merge_faqs(existing_faq, new_qa_pairs)

    save_faq(merged_faq)
    print(f"Saved updated FAQ with {len(merged_faq)} entries to {FAQ_JSON_PATH}.")

if __name__ == "__main__":
    main()
