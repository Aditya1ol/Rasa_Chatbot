from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import json
import logging
import os
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, filename="fine_tune_bert_qa.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Load dataset
dataset_path = "college_faq.json"
if not os.path.exists(dataset_path):
    logging.error(f"Dataset file {dataset_path} not found.")
    raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

if not raw_data:
    logging.error("Dataset is empty.")
    raise ValueError("Dataset is empty.")

# Prepare data for QA fine-tuning
# Expecting each item to have 'question' and 'answer' fields, and 'context' if available
# If context not available, use answer as context (simple fallback)

def prepare_features(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = []
    answers = []
    # examples is a dict of lists, so iterate by index
    for i in range(len(examples["question"])):
        context = ""
        if "context" in examples and len(examples["context"]) > i:
            context = examples["context"][i]
        if not context:
            if "answer" in examples and len(examples["answer"]) > i:
                context = examples["answer"][i]
        contexts.append(context)
        answers.append(examples["answer"][i])

    # Tokenize inputs
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        context = contexts[sample_index]

        # If no answer, set start and end to cls_index
        if answer == "" or answer is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Find start and end character index of answer in context
        start_char = context.find(answer)
        end_char = start_char + len(answer)

        # Find start and end token indices
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer not fully inside the context, set to cls_index
        if not (start_char >= 0 and end_char <= len(context)):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Otherwise find the token start and end positions
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# Convert raw data to Dataset
dataset = Dataset.from_list(raw_data)

# Tokenize and prepare features
tokenized_dataset = dataset.map(prepare_features, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="actions/trained_bert_qa",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="actions/logs",
    logging_steps=100,
    evaluation_strategy="no",
    learning_rate=3e-5,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model
try:
    trainer.train()
    logging.info("BERT QA training completed successfully")
except Exception as e:
    logging.error(f"BERT QA training failed: {e}")
    raise

# Save model and tokenizer
model.save_pretrained("actions/trained_bert_qa")
tokenizer.save_pretrained("actions/trained_bert_qa")
logging.info("BERT QA model and tokenizer saved to actions/trained_bert_qa")
