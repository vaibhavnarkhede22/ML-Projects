import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Car Price Prediction System\car data.csv")
dataset.head()

dataset.shape

dataset.describe()

dataset.info()

dataset.isnull().sum()

#checking distribution of categorical data

dataset['Fuel_Type'].value_counts()
dataset['Seller_Type'].value_counts()
dataset['Transmission'].value_counts()

from sklearn.preprocessing import LabelEncoder

lc = LabelEncoder()
dataset['Fuel_Type'] = lc.fit_transform(dataset['Fuel_Type'])
dataset['Seller_Type'] = lc.fit_transform(dataset['Seller_Type'])
dataset['Transmission'] = lc.fit_transform(dataset['Transmission'])

dataset.head()
dataset.drop(['Car_Name'], axis=1, inplace=True)
dataset.head()

x = dataset.drop(['Selling_Price'], axis=1)
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=2)

#model training

reg = LinearRegression()
reg.fit(X_train, y_train)

#model Evaluation on training data

y_pred = reg.predict(X_train)

error_score0 = metrics.r2_score(y_train, y_pred)
print("R Sqaured Error:", error_score0)

#visulaize the proces and Predicted Prices
import matplotlib.pyplot as plt

plt.scatter(y_train, y_pred, color='blue')
# plt.plot(X_train, reg.predict(X_train), color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

#model Evaluation on test data
y_pred1 = reg.predict(X_test)

error_score1 = metrics.r2_score(y_test, y_pred1)
print("R Sqaured Error:", error_score1)

#visulaize the proces and Predicted Prices
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred1)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

#2.Lassso Regression

#model training

reg1 = Lasso()
reg1.fit(X_train, y_train)

#model Evaluation on training data

y_pred = reg1.predict(X_train)

error_score2 = metrics.r2_score(y_train, y_pred)
print("R Sqaured Error:", error_score2)

#visulaize the proces and Predicted Prices
import matplotlib.pyplot as plt

plt.scatter(y_train, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

#model Evaluation on test data
y_pred1 = reg1.predict(X_test)

error_score3 = metrics.r2_score(y_test, y_pred1)
print("R Sqaured Error:", error_score3)

#visulaize the proces and Predicted Prices
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred1)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
