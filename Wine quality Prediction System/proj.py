import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Wine quality Prediction System\winequality-red.csv"
)
dataset.head()

#number of rows and columns in the dataset

dataset.shape

#to know the missing values
dataset.isnull().sum()

# sttistical description of dataset

dataset.describe()

#number of values for each qulity

sns.catplot(x='quality', data=dataset, kind='count')

#volatile acidity vs quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=dataset)

#citric acid vs quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=dataset)

# find a correlation

correl = dataset.corr()

#constrcuting a heatmap to understand the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(correl,
            annot=True,
            annot_kws={'size': 8},
            cbar=True,
            fmt='.1f',
            square=True,
            cmap='Blues')

# data Preprocessing
dataset.shape
x = dataset.iloc[:, :-1].values

#label Binarization

y = dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
print(x.shape, X_train.shape, X_test.shape)

#model fitting

cls = RandomForestClassifier()
cls.fit(X_train, y_train)

#model Evaluation
y_pred0 = cls.predict(X_train)
score0 = accuracy_score(y_train, y_pred0)
print("Accuracy score of model on training data:", score0 * 100, "%")

#model Evaluation
y_pred = cls.predict(X_test)
score1 = accuracy_score(y_test, y_pred)
print("Accuracy score of model on test data:", score1 * 100, "%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm

#making the prediction system
# input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)
input_data = (8.5, 0.28, 0.56, 1.8, 0.092, 35.0, 103.0, 0.9969, 3.3, 0.75,
              10.5)

input_arr = np.asarray(input_data)
input_data_reshaped = input_arr.reshape(1, -1)

predi = cls.predict(input_data_reshaped)
print(predi)

if (predi[0] == 1):
    print("Good Qulaity Wine")
else:
    print("Bad Quality Wine")