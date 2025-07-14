# prepare_data.py
import json
from sklearn.model_selection import train_test_split

with open("college_faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

inputs = [f"question: {item['question']}" for item in data]
targets = [item['answer'] for item in data]

train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.1)

with open("train.txt", "w", encoding="utf-8") as f:
    for i, t in zip(train_inputs, train_targets):
        f.write(f"{i}\t{t}\n")

with open("val.txt", "w", encoding="utf-8") as f:
    for i, t in zip(val_inputs, val_targets):
        f.write(f"{i}\t{t}\n")
