from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, filename="fine_tune_t5.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset
dataset_path = "data/uiet_qa.json"
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

# Validate dataset structure
for item in raw_data:
    if not all(key in item for key in ["context", "question", "answer"]):
        logging.error(f"Invalid dataset entry: {item}")
        raise ValueError(f"Invalid dataset entry: {item}")

# Convert to Dataset object
dataset = Dataset.from_list(raw_data)
logging.info(f"Loaded dataset with {len(dataset)} examples")

# Preprocess dataset for batched inputs
def preprocess_data(examples):
    # Handle batched inputs (examples is a dict of lists)
    inputs = [f"Context: {c} Question: {q} Answer:" for c, q in zip(examples["context"], examples["question"])]
    targets = examples["answer"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Tokenize targets (labels)
    labels = tokenizer(
        targets,
        max_length=150,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Remove tensor format for Trainer compatibility
    return {
        "input_ids": model_inputs["input_ids"].tolist(),
        "attention_mask": model_inputs["attention_mask"].tolist(),
        "labels": labels["input_ids"].tolist()
    }

# Apply preprocessing
try:
    tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["context", "question", "answer"])
    logging.info("Dataset preprocessing completed")
except Exception as e:
    logging.error(f"Failed to preprocess dataset: {e}")
    raise

# Training arguments
training_args = TrainingArguments(
    output_dir="actions/trained_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_dir="actions/logs",
    logging_steps=100,
    evaluation_strategy="no"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train model
try:
    trainer.train()
    logging.info("Training completed successfully")
except Exception as e:
    logging.error(f"Training failed: {e}")
    raise

# Save model and tokenizer
model.save_pretrained("actions/trained_model")
tokenizer.save_pretrained("actions/trained_model")
logging.info("Model and tokenizer saved to actions/trained_model")