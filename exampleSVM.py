#Download the spam dataset to your PC https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
#uncomment the next two lines and run them to upload the csv to /content.
#from google.colab import files
#files.upload()

#

import sklearn
import pandas as pd
import numpy as np
df= pd.read_csv("D:/2024/NCI/Semester 3/Practicum 2/GitHub/BERT test/BERTSentiment/spam.csv",encoding = "latin-1")
df= df[['v1', 'v2']]
df= df.rename(columns = {'v1': 'label', 'v2': 'text'})
#df.info()
#df.head()
# # 2. Cleaning data
#1. Removing Punctuation
import string
string.punctuation
def remove_punctuation(txt):
    txt_nopunct="".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct
df['newtext']=df['text'].apply(lambda x: remove_punctuation(x))
#Show top 10 messages with puncuation removed. Output is a , seperated list
df.head(10)['newtext'].values
#df.head()

#2. tokenisation of data
import re
def tokenize(txt):
    tokens=re.split('\W+', txt)
    return tokens
#convert to lower case
df['token_text']=df['newtext'].apply(lambda x: tokenize(x.lower()))
df.head()

#3. remove stop words
import nltk
nltk.download('stopwords')
stopwords=nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenize):
    clean=[word for word in tokenize if word not in stopwords]
    return clean
df['stop_clean']=df['token_text'].apply(lambda x: remove_stopwords(x))
#df.head()

#4. stemming
from nltk.stem import PorterStemmer
ps=PorterStemmer()
def stemming(txt):
    words=[ps.stem(word) for word in txt]
    return words
df['stem_words']=df['stop_clean'].apply(lambda x: stemming(x))
df.head()
#5. lemmatisation
def lammatization(txt):
    lam=[wn.lammetize(word) for word in txt]
    return lam
df['lam_words']=df['stem_words'].apply(lambda x: stemming(x))
df.head()
#as our "lam_words column is column of lists, and not text.
#Tfidf Vectoriser works on text so convert this column into string"
df['lam_words']=[" ".join(review) for review in df['lam_words'].values]

# # 3. split data into sets
from sklearn import model_selection, naive_bayes, svm
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['lam_words'],df['label'],test_size=0.2)
# # 4. encoding
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
# # 5. Word Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['lam_words'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)
# # 6. Use the ML Algorithms to Predict the outcome

# 1.Naive Bayes Classifier
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
# 2. Support Vector Machine

# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
#Lets try some test data
test = ["free entry"]
X_test=Tfidf_vect.transform(test)
SVM.predict(X_test)
#test1 = ["is that how you spell his name"]
#X_test1=Tfidf_vect.transform(test1)
#SVM.predict(X_test1)


