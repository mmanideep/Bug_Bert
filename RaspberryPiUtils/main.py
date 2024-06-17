from fastapi import FastAPI, HTTPException
from transformers import BertTokenizer, BertForSequenceClassification
import time
import csv
import torch

app = FastAPI()


# Running inside docker
model_path = "/app/model.pth"
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Load the model on CPU and then move to the appropriate device
model = torch.load(model_path, map_location=torch.device('cpu'))

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the selected device
model = model.to(device)


# Define a POST route for prediction
@app.post("/predict/")
async def predict(input_data: dict):
    try:
        text = input_data.get("text")

        # Check if "text" field is provided
        if text is None:
            raise HTTPException(status_code=422, detail="Field 'text' is required")

        # Prepare text input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Make prediction
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        with open("metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([len(text), inference_time])

        # Get predicted label
        predicted_class_idx = outputs.logits.argmax().item()
        
        response = True
        if predicted_class_idx == 0:
            response = False

        return {"isClassifiedAsBug": response}
    except Exception as e:
        # Return HTTP 500 Internal Server Error with error message
        raise HTTPException(status_code=500, detail=str(e))
