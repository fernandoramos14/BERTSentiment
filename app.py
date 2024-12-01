from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the trained SVM model
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define the input data structure
class Message(BaseModel):
    text: str

# Function to generate BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding.numpy()

@app.post("/classify")
async def classify_message(message: Message):
    # Get BERT embeddings for the input text
    embedding = get_bert_embeddings(message.text)
    embedding = np.array([embedding])  # Reshape for SVM input

    # Predict using the SVM model
    prediction = svm_model.predict(embedding)

    # Map the prediction to a label
    label = "spam" if prediction[0] == 1 else "ham"

    # Return the result as JSON
    return {"message": message.text, "classification": label}
