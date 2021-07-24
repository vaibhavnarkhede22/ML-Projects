import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Rock vs Mines  detection for submarine\sonar.csv",
    header=None)
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset[60].value_counts()
      )  # to check the balance of dataset  M ---->Mines R----->Rock

print(dataset.groupby(60).mean())

#splitting the dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 60].values

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=1)
print(x.shape, x_train.shape, x_test.shape)

#Traning the model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

#evaluate the Model using accuracy score
#acuracy on training data
y_pred = log_reg.predict(x_train)

# score=accuracy_score(y_train,log_reg.predict(x_train))
score = accuracy_score(y_train, y_pred)
print("Accuracy on Training data is:", score)

#evaluate the Model using accuracy score
#acuracy on testingdata
y_pred = log_reg.predict(x_test)

# score=accuracy_score(y_train,log_reg.predict(x_train))
score = accuracy_score(y_test, y_pred)
print("Accuracy on Testing data is:", score)

#making the predictive system
# input_data =(0.0079,0.0086,0.0055,0.0250,0.0344,0.0546,0.0528,0.0958,0.1009,0.1240,0.1097,0.1215,0.1874,0.3383,0.3227,0.2723,0.3943,0.6432,0.7271,0.8673,0.9674,0.9847,0.9480,0.8036,0.6833,0.5136,0.3090,0.0832,0.4019,0.2344,0.1905,0.1235,0.1717,0.2351,0.2489,0.3649,0.3382,0.1589,0.0989,0.1089,0.1043,0.0839,0.1391,0.0819,0.0678,0.0663,0.1202,0.0692,0.0152,0.0266,0.0174,0.0176,0.0127,0.0088,0.0098,0.0019,0.0059,0.0058,0.0059,0.0032)
input_data = (0.0968, 0.0821, 0.0629, 0.0608, 0.0617, 0.1207, 0.0944, 0.4223,
              0.5744, 0.5025, 0.3488, 0.1700, 0.2076, 0.3087, 0.4224, 0.5312,
              0.2436, 0.1884, 0.1908, 0.8321, 1.0000, 0.4076, 0.0960, 0.1928,
              0.2419, 0.3790, 0.2893, 0.3451, 0.3777, 0.5213, 0.2316, 0.3335,
              0.4781, 0.6116, 0.6705, 0.7375, 0.7356, 0.7792, 0.6788, 0.5259,
              0.2762, 0.1545, 0.2019, 0.2231, 0.4221, 0.3067, 0.1329, 0.1349,
              0.1057, 0.0499, 0.0206, 0.0073, 0.0081, 0.0303, 0.0190, 0.0212,
              0.0126, 0.0201, 0.0210, 0.0041)
#changing the input data to numpy array
input_data_arr = np.asarray(input_data)

#reshape the numpy array as we are predicting for one instance
input_data_arr1 = input_data_arr.reshape(1, -1)

predi = log_reg.predict(input_data_arr1)
# print(predi)
if predi == 'R':
    print("Your Prediction is Rock")
else:
    print("Your prediction is Mine")