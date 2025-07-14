from parrot import Parrot
import torch
import json
import os

# 🧠 Enable GPU if available
USE_GPU = torch.cuda.is_available()
print(f"Using GPU: {USE_GPU}")

# 🦜 Initialize Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=USE_GPU)

# 📂 Load your JSON data
INPUT_FILE = "college_faq.json"
OUTPUT_FILE = "augmented_data.json"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File '{INPUT_FILE}' not found!")

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

augmented_data = []

for item in data:
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()

    # Add the original
    augmented_data.append({"question": question, "answer": answer})

    try:
        # Skip short questions
        if len(question.split()) < 4:
            continue

        # Paraphrase question (max 3 versions)
        paraphrases = parrot.augment(input_phrase=question, max_return_phrases=3)

        if paraphrases:
            for para in paraphrases:
                para_question = para[0].strip()
                if para_question.lower() != question.lower():
                    augmented_data.append({"question": para_question, "answer": answer})

    except Exception as e:
        print(f"⚠️ Error paraphrasing: {question}\n{e}")

# 💾 Save augmented output
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(augmented_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Augmented data saved to '{OUTPUT_FILE}' ({len(augmented_data)} total items).")
