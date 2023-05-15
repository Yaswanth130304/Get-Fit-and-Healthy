import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

diabetes_data = pd.read_csv("diabetes.csv")

diabetes_data.describe()

diabetes_data['Outcome'].value_counts()

diabetes_data.groupby('Outcome').mean()

X = diabetes_data.drop(columns = 'Outcome',axis=1)
Y = diabetes_data['Outcome']

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel = 'linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy : ",training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Testing accuracy : ',testing_data_accuracy)

filename ='Diabetes_pickle.sav'
pickle.dump(classifier,open(filename,'wb'))

file_loader = pickle.load(open(filename,'rb'))
result = file_loader.predict(X_test)
print('final result = ',accuracy_score(result, Y_test))

