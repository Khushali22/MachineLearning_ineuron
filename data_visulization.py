# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt 


data = pd.read_csv("iris.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,4].values

# Encoding categorical data
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
zs = X[:,[2]]
xs = X[:,[0]]
ys = X[:,[1]]

ax.scatter(xs, ys, zs,cmap=plt.get_cmap('spring'), edgecolor='face')
plt.show()