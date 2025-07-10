import json
from transformers import pipeline

INPUT_FAQ_PATH = "college_faq.json"
OUTPUT_PARAPHRASED_PATH = "college_faq_paraphrased.json"

def generate_paraphrases(question, num_return_sequences=3):
    paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")
    input_text = f"paraphrase: {question} </s>"
    paraphrases = paraphraser(input_text, max_length=256, num_return_sequences=num_return_sequences, num_beams=10)
    return [p['generated_text'] for p in paraphrases]

def main():
    with open(INPUT_FAQ_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    augmented_faq = []
    for entry in faq_data:
        question = entry["question"]
        answer = entry["answer"]
        paraphrases = generate_paraphrases(question)
        # Add original question
        augmented_faq.append({"question": question, "answer": answer})
        # Add paraphrased questions
        for para in paraphrases:
            augmented_faq.append({"question": para, "answer": answer})

    with open(OUTPUT_PARAPHRASED_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented_faq, f, indent=2, ensure_ascii=False)

    print(f"Paraphrased FAQ saved to {OUTPUT_PARAPHRASED_PATH}")

if __name__ == "__main__":
    main()
