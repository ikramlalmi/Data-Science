from turtle import shape
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#fetching the data
df = pd.read_csv(Users/MacBook/Desktop/Data_Science_Projects/Data-Science/Parkinson’s Disease Detection/parkinsons.data)
df.shape
df.head()

#divide the clomns names into feates and labels

features = df.loc[:,lambda df: df.columns!= "status"]


