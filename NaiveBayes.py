#Naive Bayes
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator
    
    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._pdf(c, x)))
            posteriors.append(prior + class_conditional)
        
        return self.classes[np.argmax(posteriors)]

