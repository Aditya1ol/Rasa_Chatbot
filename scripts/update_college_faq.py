import os
import json
import re
from extract_pdf import extract_chunks_from_pdf

PDF_FOLDER = "pdfs"
FAQ_JSON_PATH = "college_faq.json"

def is_question(text):
    """
    Improved heuristic to detect questions:
    - Lines ending with a question mark and reasonable length
    - Lines starting with common question words
    - Lines starting with 'Question' keyword
    - Lines containing a question mark anywhere
    """
    question_words = ['what', 'when', 'where', 'who', 'whom', 'which', 'why', 'how', 'is', 'are', 'can', 'do', 'does', 'did', 'will', 'would', 'should', 'could']
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    if len(text_stripped) < 5:
        return False
    if text_lower.startswith('question '):
        return True
    if text_lower.endswith('?'):
        return True
    for qw in question_words:
        if text_lower.startswith(qw + ' '):
            return True
    if '?' in text_stripped:
        return True
    return False

import re

def parse_qa_pairs(chunks):
    """
    Improved parsing of Q&A pairs from text chunks.
    Supports:
    - 'Question X:' labels
    - 'Answer:' labels
    - Multiline answers
    - Flexible question detection using is_question()
    """
    qa_pairs = []
    current_question = None
    current_answer = []
    for chunk in chunks:
        chunk_stripped = chunk.strip()
        # Check if chunk contains a question label
        question_match = re.match(r'Question\s*\d+:\s*(.*)', chunk_stripped, re.DOTALL)
        if question_match:
            # Save previous Q&A pair if exists
            if current_question and current_answer:
                qa_pairs.append({
                    "question": current_question,
                    "answer": ' '.join(current_answer).strip()
                })
                current_answer = []
            current_question = question_match.group(1).strip()
            # Remove question label from chunk to get remaining text
            remaining_text = re.sub(r'Question\s*\d+:\s*', '', chunk_stripped, flags=re.DOTALL).strip()
            # Check if remaining text contains answer label
            answer_match = re.search(r'Answer:\s*(.*)', remaining_text, re.DOTALL)
            if answer_match:
                current_answer.append(answer_match.group(1).strip())
        else:
            # If chunk contains answer label
            answer_match = re.match(r'Answer:\s*(.*)', chunk_stripped, re.DOTALL)
            if answer_match:
                current_answer.append(answer_match.group(1).strip())
            else:
                # If chunk looks like a question and no current question, start new question
                if is_question(chunk_stripped) and not current_question:
                    current_question = chunk_stripped
                    current_answer = []
                # Append to current answer if question exists
                elif current_question:
                    current_answer.append(chunk_stripped)
                else:
                    # Ignore chunks that are neither question nor answer
                    pass
    # Save last Q&A pair
    if current_question and current_answer:
        qa_pairs.append({
            "question": current_question,
            "answer": ' '.join(current_answer).strip()
        })
    return qa_pairs

def load_existing_faq():
    if os.path.exists(FAQ_JSON_PATH):
        with open(FAQ_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_faq(faq_list):
    with open(FAQ_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(faq_list, f, indent=2, ensure_ascii=False)

def merge_faqs(existing, new):
    existing_questions = set(item['question'] for item in existing)
    merged = []
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

    new_qa_pairs = parse_qa_pairs(all_chunks)
    print(f"Parsed {len(new_qa_pairs)} new Q&A pairs.")

    existing_faq = load_existing_faq()
    merged_faq = merge_faqs(existing_faq, new_qa_pairs)

    save_faq(merged_faq)
    print(f"Saved updated FAQ with {len(merged_faq)} entries to {FAQ_JSON_PATH}.")

if __name__ == "__main__":
    main()
