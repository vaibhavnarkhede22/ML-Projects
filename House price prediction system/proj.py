import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

dataset = sklearn.datasets.load_boston()
print(dataset)

# data----->features
#target ----->price of the houses
# feature_names----->column names

boston_data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
boston_data.head()
boston_data['price'] = dataset.target
boston_data.head()
boston_data.shape
boston_data.isnull().sum()
boston_data.describe()
#understanding the correaltion between various features in the dataset
#1.Positive Correlation
#2.Negative Correlation
corr = boston_data.corr()

#constructing the heatmap to understand the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(corr,
            cbar=True,
            square=True,
            fmt='.1f',
            annot=True,
            annot_kws={'size': 8},
            cmap='Blues')

x = boston_data.iloc[:, :-1].values
y = boston_data.iloc[:, 13].values

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=2)
print(x.shape, X_train.shape, X_test.shape)

# model training

reg = XGBRegressor()
reg.fit(X_train, y_train)

#prediction on training data

y_pred = reg.predict(X_train)
print(y_pred)

# evaluation of model
# on training data

# R squared error only for regression
score1 = metrics.r2_score(y_train, y_pred)
print("R squared Error", score1)

# mean absolute error
score2 = metrics.mean_absolute_error(y_train, y_pred)
print("Mean Squared Error", score2)

# on test data

y_pred1 = reg.predict(X_test)
# R squared error only for regression
score3 = metrics.r2_score(y_test, y_pred1)
print("R squared Error", score1)

# mean absolute error
score4 = metrics.mean_absolute_error(y_test, y_pred1)
print("Mean Squared Error", score2)

#visualizing the actual and predicted prices
plt.scatter(y_train, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Value( Training Data)")
plt.show()

plt.scatter(y_test, y_pred1)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Price")
plt.title("Actual Price Vs Predicted Value( Testing Data)")
plt.show()