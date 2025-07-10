import os
import json
from extract_pdf import extract_chunks_from_pdf
from transformers import pipeline
import warnings

PDF_FOLDER = "pdfs"
GENERATED_QUESTIONS_PATH = "generated_questions.json"

def generate_questions(chunks):
    warnings.filterwarnings("ignore")
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", 
                                  do_sample=False, num_beams=5)
    questions = []
    max_input_length = 512
    for chunk in chunks:
        try:
            inputs = question_generator.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_input_length)
            input_text = question_generator.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            outputs = question_generator(input_text, max_length=64)
            for output in outputs:
                question = output.get("generated_text")
                if question and question not in questions:
                    questions.append(question)
        except Exception as e:
            print(f"Error generating questions for chunk: {e}")
    return questions

def process_pdf_file(pdf_path):
    try:
        chunks = extract_chunks_from_pdf(pdf_path)
        return chunks
    except Exception as e:
        print(f"Failed to process {pdf_path}: {e}")
        return []

def main():
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing {filename}...")
            chunks = process_pdf_file(pdf_path)
            all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} chunks from PDFs.")

    questions = generate_questions(all_chunks)
    print(f"Generated {len(questions)} questions.")

    with open(GENERATED_QUESTIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved generated questions to {GENERATED_QUESTIONS_PATH}.")

if __name__ == "__main__":
    main()
