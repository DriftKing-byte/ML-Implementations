# Decision Tree

import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if depth >= self.max_depth or n_labels == 1:
            return self._most_common_label(y)
        
        best_feat, best_thresh = self._best_split(X, y)
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = X[:, best_feat] >= best_thresh
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return (best_feat, best_thresh, left, right)
    
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feature], thresh)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh
        
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        
        left_idxs = X_column < split_thresh
        right_idxs = X_column >= split_thresh
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        
        child_entropy = (n_l/n)*self._entropy(y[left_idxs]) + (n_r/n)*self._entropy(y[right_idxs])
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return [self._traverse(x, self.tree) for x in X]
    
    def _traverse(self, x, node):
        if not isinstance(node, tuple):
            return node
        
        feature, thresh, left, right = node
        if x[feature] < thresh:
            return self._traverse(x, left)
        return self._traverse(x, right)

