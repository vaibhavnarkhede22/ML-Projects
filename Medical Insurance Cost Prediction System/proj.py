import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#loading the dataset
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Medical Insurance Cost Prediction System\insurance.csv"
)
dataset.head()
dataset.tail()
#numbe of rows and columns
dataset.shape

#getting some information about the dataset
dataset.info()

## categorical features
# sex, smoker ,region
dataset.isnull().sum()

# data analysis
dataset.describe()

#dirstribution of age value
sns.set()
plt.figure(figsize=(6, 6))
sns.distplot(dataset['age'])
plt.title("Age Distribution")
plt.show()

#gender column
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=dataset)
plt.title("Sex Distribution")
plt.show()

dataset['sex'].value_counts()

#bmi distribution
plt.figure(figsize=(6, 6))
sns.distplot(dataset['bmi'])
plt.title("bmi distribution")
plt.show()

#normal bmi range--->18.5 to 24.9

#childern column
plt.figure(figsize=(6, 6))
sns.countplot(x='children', data=dataset)
plt.title("Children")
plt.show()

dataset['children'].value_counts()

# smoker Column
plt.figure(figsize=(6, 6))
sns.countplot(x='smoker', data=dataset)
plt.title("Smoker")
plt.show()

dataset['smoker'].value_counts()

#region column
plt.figure(figsize=(6, 6))
sns.countplot(x='region', data=dataset)
plt.title("region")
plt.show()

dataset['region'].value_counts()

#distribution of charges

plt.figure(figsize=(6, 6))
sns.distplot(dataset['charges'])
plt.title("Charges Distribution")
plt.show()

#data Preprocessing
from sklearn.preprocessing import LabelEncoder

lc = LabelEncoder()
dataset['sex'] = lc.fit_transform(dataset['sex'])
dataset['smoker'] = lc.fit_transform(dataset['smoker'])
dataset['region'] = lc.fit_transform(dataset['region'])

#splitting the features and target
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

#splitting the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=2)
print(x.shape, X_train.shape, X_test.shape)
#model training

reg = LinearRegression()
reg.fit(X_train, y_train)

#model evaluation
# for training data
y_pred = reg.predict(X_train)
score0 = metrics.r2_score(y_train, y_pred)
print("R Sqaured Error on Training data:", score0 * 100, "%")

# for test data
y_pred1 = reg.predict(X_test)
score1 = metrics.r2_score(y_test, y_pred1)
print("R Sqaured Error on Test data:", score1 * 100, "%")

#buliding the predictive system
input_data = ()
input_arr = np.asarray(input_data)
input_arr_re = input_arr.reshape(1, -1)

#prediting for new values
predi = reg.predict(input_arr_re)
print("The insurance cost in USD is:", predi[0])
