from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import Mapping, Dict
import time
import json

# Load saved model and tokenizer
model_path = "D:\\AI and ML Masters\\BugBert\\BugBertModel\\model.pth"
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = torch.load(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare text input
text = "I can't use the AbstractHttpProcessor as it is for asynchronously\nprocessing different requests, because it is hard-wired to use a\nsingle context which can not be changed. Async requires different\ncontexts for requests. Patch follows.\n\ncheers,\n  Roland"  # Example
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted label
predicted_class_idx = outputs.logits.argmax().item()

print(f"Prediction: {predicted_class_idx}")



