# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
y_hc_lin = hc.fit_predict(X)

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='average')
y_hc_avg = hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='clus1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='clus2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='cyan',label='clus3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='magenta',label='clus4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='green',label='clus5')
plt.show()

plt.scatter(X[y_hc_lin==0,0],X[y_hc_lin==0,1],s=100,c='red',label='clus1')
plt.scatter(X[y_hc_lin==1,0],X[y_hc_lin==1,1],s=100,c='blue',label='clus2')
plt.scatter(X[y_hc_lin==2,0],X[y_hc_lin==2,1],s=100,c='cyan',label='clus3')
plt.scatter(X[y_hc_lin==3,0],X[y_hc_lin==3,1],s=100,c='magenta',label='clus4')
plt.scatter(X[y_hc_lin==4,0],X[y_hc_lin==4,1],s=100,c='green',label='clus5')
plt.show()

plt.scatter(X[y_hc_avg==0,0],X[y_hc_avg==0,1],s=100,c='red',label='clus1')
plt.scatter(X[y_hc_avg==1,0],X[y_hc_avg==1,1],s=100,c='blue',label='clus2')
plt.scatter(X[y_hc_avg==2,0],X[y_hc_avg==2,1],s=100,c='cyan',label='clus3')
plt.scatter(X[y_hc_avg==3,0],X[y_hc_avg==3,1],s=100,c='magenta',label='clus4')
plt.scatter(X[y_hc_avg==4,0],X[y_hc_avg==4,1],s=100,c='green',label='clus5')
plt.show()

from sklearn.metrics import adjusted_rand_score
ward_ar_score = adjusted_rand_score(y_hc,y_hc)

from sklearn.metrics import adjusted_rand_score
ward_ar_score_avg = adjusted_rand_score(y_hc,y_hc_avg)

from sklearn.metrics import adjusted_rand_score
ward_ar_score_com = adjusted_rand_score(y_hc,y_hc_lin)

from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

plt.scatter(normalized_X[:,0],normalized_X[:,1],color='red')
plt.show()

from sklearn.preprocessing import normalize
normalized_X1 = normalize(X)

from scipy.cluster.hierarchy import linkage
linkage_type = 'ward'

linkage_matrix = linkage(X,linkage_type)

from scipy.cluster.hierarchy import dendrogram
dendrogram = dendrogram(linkage_matrix)
plt.show()

import seaborn as sns
sns.clustermap(X,figsize=(18,50),method='ward',cmap='viridis')
plt.show()


#DBSCAN
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

import dbscan_lab_helper as helper
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1,min_samples=3)
ypred_dbscan = dbscan.fit_predict(X)


plt.scatter(X[ypred_dbscan==-1,0],X[ypred_dbscan==-1,1],s=100,c='red',label='clus1')
plt.scatter(X[ypred_dbscan==1,0],X[ypred_dbscan==1,1],s=100,c='blue',label='clus2')
plt.scatter(X[ypred_dbscan==2,0],X[ypred_dbscan==2,1],s=100,c='cyan',label='clus3')
plt.scatter(X[ypred_dbscan==3,0],X[ypred_dbscan==3,1],s=100,c='magenta',label='clus4')
plt.scatter(X[ypred_dbscan==4,0],X[ypred_dbscan==4,1],s=100,c='green',label='clus5')
plt.show()

