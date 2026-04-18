#KMeans

import numpy as np

class KMeans:
    def __init__(self, k=3, n_iters=100):
        self.k = k
        self.n_iters = n_iters
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # initialize centroids randomly
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]
        
        for _ in range(self.n_iters):
            clusters = self._create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self._get_centroids(X, clusters)
            
            if np.all(old_centroids == self.centroids):
                break
    
    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        
        return clusters
    
    def _closest_centroid(self, sample):
        distances = [np.linalg.norm(sample - point) for point in self.centroids]
        return np.argmin(distances)
    
    def _get_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))
        
        for idx, cluster in enumerate(clusters):
            cluster_points = X[cluster]
            centroids[idx] = np.mean(cluster_points, axis=0)
        
        return centroids
    
    def predict(self, X):
        return [self._closest_centroid(x) for x in X]
