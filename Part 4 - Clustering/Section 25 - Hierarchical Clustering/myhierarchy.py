import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c = 'red',label='first')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c = 'blue',label='second')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c = 'green',label='third')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c = 'cyan',label='fourth')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c = 'magenta',label='fifth')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c = 'yellow',label='centroids')
plt.show()