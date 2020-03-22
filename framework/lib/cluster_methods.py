from sklearn import manifold #needed for multidimensional scaling (MDS) and t-SNE
from sklearn import cluster #needed for k-Means clustering
from sklearn import preprocessing #needed for scaling attributes to the nterval [0,1]
import numpy as np
import os
import matplotlib.pyplot as plt


colors = np.array(['orange', 'blue', 'lime', 'blue', 'khaki', 'pink', 'green', 'purple'])

# points - a 2D array of (x,y) coordinates of data points
# labels - an array of numeric labels in the interval [0..k-1], one for each point
# centers - a 2D array of (x, y) coordinates of cluster centers
# title - title of the plot


def clustering_scatterplot(points, labels, centers, title):
    
    
    n_clusters = np.unique(labels).size
    for i in range(n_clusters):
        h = plt.scatter(points[labels==i,0],
                        points[labels==i,1], 
                        c=colors[i%colors.size],
                        label = 'cluster '+str(i))

    # plot the centers of the clusters
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c='r', marker='*', s=500)

    _ = plt.title(title)
    _ = plt.legend()
    _ = plt.xlabel('x')
    _ = plt.ylabel('y')


def save_cluster(k, df, cluster_path, name):
  clustered_data_sklearn = cluster.KMeans(n_clusters=k, n_init=10, max_iter=300).fit(df)
  data_and_centers = np.r_[df,clustered_data_sklearn.cluster_centers_]
  XYcoordinates = manifold.MDS(n_components=2).fit_transform(data_and_centers)
  print("transformation complete")
  clustering_scatterplot(points=XYcoordinates[:-k,:], 
                        labels=clustered_data_sklearn.labels_, 
                        centers=XYcoordinates[-k:,:], 
                        title='MDS')
  cluster_file = os.path.join(cluster_path, name + '.png')
  plt.savefig(cluster_file)

