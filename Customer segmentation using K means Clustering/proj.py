import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#loading the dataset
dataset = pd.read_csv(
    "D:\Python Doc\ML Projects\Customer segmentation using K means Clustering\Mall_Customers.csv"
)
dataset.head()

#finding the number of rows and column
dataset.shape
# getting some more info about dataset

dataset.info()

# getting some statistical information
dataset.describe()

#to know the missing values
dataset.isnull().sum()

#choosing the annual income column and spending score column
x = dataset.iloc[:, [3, 4]].values
print(x)

#choosing the number of clutsers
#wcss--->within clusters  sum of squares
# Elbow Method to find number of clutsers
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

#plot an elbow graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

# the optimum no of clusters are 5 as per elbow method
# training the k Means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

#return a label for each data point based on their cluster
y = kmeans.fit_predict(x)
print(y)

# visualizing all the clusters
#plotting all the clusters and their centroids

# 5 clusters as= 0,1,2,3,4

plt.figure(figsize=(8, 8))
plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s=50, c='blue', label='Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s=50, c='cyan', label='Cluster 5')

#plotting the centroids

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100,
            color='black',
            label='Centroid')
plt.title("Customer Groups")
plt.xlabel("Annual Income")
plt.ylabel('Spending score')
plt.show()