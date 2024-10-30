import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load the data
df = pd.read_csv('D:/2024/NCI/Semester 3/Practicum 2/GitHub/BERT test/BERTSentiment/spam.csv',encoding = 'latin-1')


# Check the data
print(df.head())

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

# Example to see the embeddings
print(df[['v2', 'embeddings']].head())
