import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# data processing
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Loan status Prediction System\\train_u6lujuX_CVtuZ9i (1).csv"
)
dataset.head()

print(dataset.shape)

# statistical measures
dataset.describe()

#number of missing values in columns
dataset.isnull().sum()

#dropping the missing values
dataset.dropna(inplace=True)
dataset.isnull().sum()

#label encoding
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
dataset['Loan_Status'] = enc.fit_transform(dataset['Loan_Status'])
dataset.head()
dataset['Gender'] = enc.fit_transform(dataset['Gender'])
dataset['Married'] = enc.fit_transform(dataset['Married'])
dataset['Education'] = enc.fit_transform(dataset['Education'])
dataset['Self_Employed'] = enc.fit_transform(dataset['Self_Employed'])
dataset['Property_Area'] = enc.fit_transform(dataset['Property_Area'])
dataset.head()

#dependant values
dataset['Dependents'].value_counts()
dataset['Property_Area'].value_counts()
#replacing the value of 3+  as 4
dataset = dataset.replace(to_replace='3+', value=4)
dataset['Dependents'].value_counts()

#visualizing the data
#education and loan status
sns.countplot(x='Education', hue='Loan_Status', data=dataset)

#marrital status and loan Status
sns.countplot(x='Married', hue='Loan_Status', data=dataset)

x = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=2)
print(x.shape, X_train.shape, X_test.shape)

# Training the model SVM
cls = svm.SVC(kernel='linear')
cls.fit(X_train, y_train)

#model evaluation

#accuracy score on training data
y_pred = cls.predict(X_train)
score1 = accuracy_score(y_train, y_pred)
print("Accuracy on Training data", score1 * 100, "%")

#accuracy score on test data
y_pred1 = cls.predict(X_test)
score2 = accuracy_score(y_test, y_pred1)
print("Accuracy on Testing data", score2 * 100, "%")

#making the predictive system
# input_data = (1, 1, 1, 1, 0, 4583, 1508, 128, 360, 1, 1)
input_data = (1, 1, 1, 1, 0, 12841, 10968, 349, 360, 1, 2)
input_arr = np.asarray(input_data)
input_arr_re = input_arr.reshape(1, -1)

predi = cls.predict(input_arr_re)

print(predi)
if predi[0] == 0:
    print("Loan not Approved")
else:
    print("Loan Approved")
