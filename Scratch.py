import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('.\\TrainingData\\1.csv', delimiter='\t', index_col=0)

X = df.X.values
Y = df.Y.values

train = list(zip(X,Y**8))
kmeans = KMeans(n_clusters=20, random_state=0).fit(train)

plt.scatter(X,Y,c=kmeans.labels_)
# plt.plot(X,Y)
plt.show()

