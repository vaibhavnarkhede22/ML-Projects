import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

#loading the dataset
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Diabetes_Prediction\diabetes.csv")
print(dataset.head())

#1---->diabetic 0---->Non diabetic

# number of rows and columns here in dataset
print(dataset.shape)

#getting the statistical measures of the data
dataset.describe()

# getting to know how many are there as diabetic and non diabetic persons from dataset

dataset['Outcome'].value_counts()

dataset.groupby('Outcome').mean()

#splitting the data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=2)

# data standardization

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(x_train.shape, x_test.shape)

# training the model

cls = SVC(kernel='linear')
cls.fit(x_train, y_train)

# evaluate the model
#accuracy score on the training data
y_pred = cls.predict(x_test)
score = accuracy_score(y_train, cls.predict(x_train))
print("Accuracy score on the training data", score * 100, "%")

score = accuracy_score(y_test, y_pred)
print("Accuracy score on the test data", score * 100, "%")

clr = classification_report(y_test, y_pred)
print(clr)

# Making a prediction system
# input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)

# input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
# input_data = (1, 122, 90, 51, 220, 49.7, 0.325, 31)
input_data = (7, 160, 54, 32, 175, 30.5, 0.588, 39)

input_data_arr = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance
input_data_arr1 = input_data_arr.reshape(1, -1)

data_std = sc_x.transform(input_data_arr1)
print(data_std)

predi = cls.predict(data_std)
# predi = cls.predict(input_data_arr1)
print(predi)
if predi[0] == 0:
    print("The person is non Diabetic")
else:
    print("The Person is Diabetic")