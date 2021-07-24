import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Gold Price Prediction system\gld_price_data.csv"
)
dataset.head()

#to know the number of rows and columns in the dataset
dataset.shape

#getting the info
dataset.info()

dataset.isnull().sum()

dataset.describe()

# getting the correlation in the data

corr = dataset.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr,
            cmap='Blues',
            annot=True,
            cbar=True,
            square=True,
            fmt='.1f',
            annot_kws={'size': 8})

print(corr['GLD'])

#checking the distribution plot

sns.distplot(dataset['GLD'], color='green')

#splitting the features and target

x = dataset.drop(['Date', 'GLD'], axis=1)
y = dataset['GLD']

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=2,
                                                    test_size=0.2)
#model training
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

#model Evaluation

y_pred = reg.predict(X_test)
print(y_pred)

from sklearn import metrics

err = metrics.r2_score(y_test, y_pred)
print("R sqaured error is:", err)

#visualize the results
y_test = list(y_test)
plt.plot(y_test, color='red', label="Actual Value")
plt.plot(y_pred, color='green', label="Predicted Value")

plt.xlabel("Number of values")
plt.ylabel("GLD Price")
plt.title("Actual Price vs Predicted price")
plt.legend()
plt.show()