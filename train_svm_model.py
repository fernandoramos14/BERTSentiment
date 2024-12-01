from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.stem import PorterStemmer
import re
import torch
from transformers import BertTokenizer, BertModel
import pickle


file_path = "D:/2024/NCI/Semester 3/Practicum 2/GitHub/BERT test/BERTSentiment/spam_.csv" 
df = pd.read_csv(file_path, encoding="latin-1")
#df.head()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings
def get_bert_embeddings(text):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get the BERT output
    with torch.no_grad():
        outputs = model(**inputs)

    # Take the embedding of the [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding

# Generate embeddings for each text in the dataframe
df['embeddings'] = df['v2'].apply(get_bert_embeddings)


# Convert labels to numerical values (e.g., "ham" -> 0, "spam" -> 1)
df['label'] = df['v1'].apply(lambda x: 1 if x == 'spam' else 0)

# Convert embeddings to a 2D array format
X = np.array([embedding.numpy() for embedding in df['embeddings']])
y = df['label'].values

#print(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced' )
svm_model.fit(X_train, y_train)

print("SVM model trained successfully!")


# Generates model
def generate_svm():
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    print("SVM model saved to 'svm_model.pkl'")

generate_svm()
