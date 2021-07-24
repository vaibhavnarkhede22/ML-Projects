import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading of the dataset
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Heart Disease PRediction System\heart.csv")
dataset.head()

#getting the shape of dataset
dataset.shape
dataset.describe()
dataset.info()
dataset.isnull().sum()
#checking the distribution of target variable

dataset['target'].value_counts()

#1---->defective heart
#0----> Healthy Heart

#splitting the input features and labels

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values
print(x)
print(y)

#splitting data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=2)
print(x.shape, X_train.shape, X_test.shape)

#model Training

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#model Evaluation
#accuracy on training data
y_pred0 = log_reg.predict(X_train)
score0 = accuracy_score(y_train, y_pred0)
print("Accuracy Score on training data is :", score0 * 100, "%")

#accuracy on testing data
y_pred1 = log_reg.predict(X_test)
score1 = accuracy_score(y_test, y_pred1)
print("Accuracy Score on test data is :", score1 * 100, "%")

# making the prediction system

# input_data = (34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2)
input_data = (58, 1, 2, 112, 230, 0, 0, 165, 0, 2.5, 1, 1, 3)

input_data1 = np.asarray(input_data)
input_d2 = input_data1.reshape(1, -1)

predi = log_reg.predict(input_d2)
print(predi)
if predi[0] == 0:
    print("The person is not having any heart disease")
else:
    print("The person has heart disease")
