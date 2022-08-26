from cgitb import text
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#creating the DF
file_path = "/Users/MacBook/Downloads/news.zip"
df = pd.read_csv(file_path)

#checking the shape of the DF

df.head(20)
df.shape

#Getting the labels from the data frame.

labels = df.label
print(labels)

#splitting the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size= 0.2, random_state = 7)

#initialize the TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7

vectorizer = TfidfVectorizer(max_df=0.7, stop_words="english")

train_tfidf = vectorizer.fit_transform(x_train)
test_tfidf = vectorizer.transform(x_test)

vectorizer.get_feature_names_out()

#initialize the PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(max_iter = 50)

# Fitting model
pac.fit(train_tfidf, y_train)

# Making prediction on test set
y_pred = pac.predict(test_tfidf)

# Model evaluation
score = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy : {round(score * 100, 2)} %\n\n")  

print( f"Classification Report : \n\n{classification_report(y_test, y_pred)}")

#confusion matrix
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))