import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Credit card Fraud Detection\creditcard.csv")
dataset.head()

dataset.tail()
#getting the inforamtion about the dataset
dataset.info()
#getting the statistical data from dataset
dataset.describe()

#checking for any null values
dataset.isnull().sum()

#checking the distibution of transactions

dataset['Class'].value_counts()

# the dataset is highly unbalanced

# 0-----Normal transaction
# 1-----fraud Transaction

# separating the data for analysis
normal = dataset[dataset.Class == 0]
fraud = dataset[dataset.Class == 1]

print(normal.shape)
print(fraud.shape)

#statistical Measures of the data
normal.Amount.describe()

fraud.Amount.describe()

#comapre the values for both transactions

dataset.groupby('Class').mean()

#undersampling
#build a sample dataset containing the similar distribution of normal and fraud transactions

# no of fraud Transactions---->492

normal_sample = normal.sample(n=492)

#concatenate two frames
new_dataset = pd.concat([normal_sample, fraud], axis=0)
new_dataset.head()

new_dataset.tail()
new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#splitting the data into features and Labels

x = new_dataset.iloc[:, :-1].values
y = new_dataset.iloc[:, 30].values

print(x)
print(y)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=2)
print(x.shape, X_train.shape, X_test.shape)

#model training
log = LogisticRegression()
log.fit(X_train, y_train)

#model Evaluation

# Accuracy Score on training data

y_pred0 = log.predict(X_train)
score0 = accuracy_score(y_train, y_pred0)
print("Accuarcy score on training data:", score0 * 100, "%")

# Accuracy Score on test data

y_pred1 = log.predict(X_test)
score1 = accuracy_score(y_test, y_pred1)
print("Accuarcy score on test data:", score0 * 100, "%")
