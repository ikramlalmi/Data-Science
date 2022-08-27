import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#fetching the data
file_path = "/Users/MacBook/Desktop/Data_Science_Projects/Data-Science/Parkinsons_ Disease_Detection/parkinsons.data"
df = pd.read_csv(file_path)
df.shape
# print(df.head())

#divide the columns names into features and labels
features = df.loc[:,lambda df: df.columns!= "status"].values[:,1:]# taking all rows and dropping the name column
labels = df.loc[:,"status"].values
#print(features)

#counting the numbers of rows with and without Parkinson's
# print(len(labels[labels == 1]))
print(f"{labels[labels == 1].shape[0]},{labels[labels == 0].shape[0]}")

#Scale the features to be between -1 and 1
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

#split the dataset into training and testing sets keeping 20% of the data for testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 7)

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Model evaluation
score = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy : {round(score * 100, 2)} %\n\n")

print(f"Classification Report : \n\n{classification_report(y_test, y_pred)}")

#confusion matrix
print(confusion_matrix(y_test, y_pred))

