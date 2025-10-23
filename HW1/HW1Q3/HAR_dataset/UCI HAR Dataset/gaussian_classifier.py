import numpy as np
from scipy.linalg import pinv, det

class GaussianClassifier:
    """Gaussian classifier with regularization for covariance matrices"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.classes = None
        self.priors = {}
        self.means = {}
        self.covariances = {}
        self.regularized_covariances = {}

    def eval_gaussian_pdf(self, x, mu, Sigma):
        """Evaluate Gaussian probability density function"""
        n = x.shape[0]
        
        C = (2 * np.pi) ** (-n / 2) * det(Sigma) ** (-0.5)
        a = x - mu.reshape(-1, 1)
        b = pinv(Sigma) @ a
        
        px = C * np.exp(-0.5 * np.sum(a * b, axis=0))
        return px

    def fit(self, X, y):
        """Train classifier by estimating class parameters from data"""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for class_label in self.classes:
            class_mask = (y == class_label)
            X_class = X[class_mask]
            n_class = X_class.shape[0]
            
            self.priors[class_label] = n_class / n_samples
            self.means[class_label] = np.mean(X_class, axis=0)
            
            X_centered = X_class - self.means[class_label]
            self.covariances[class_label] = (X_centered.T @ X_centered) / n_class
            
            lambda_reg = self._compute_regularization(self.covariances[class_label])
            self.regularized_covariances[class_label] = (
                self.covariances[class_label] + lambda_reg * np.eye(n_features)
            )
    
    def _compute_regularization(self, covariance_matrix):
        """Calculate regularization parameter to stabilize covariance matrix"""
        try:
            trace_C = np.trace(covariance_matrix)
            rank_C = np.linalg.matrix_rank(covariance_matrix)
            if rank_C == 0:
                return self.alpha * 100
            lambda_reg = self.alpha * trace_C / rank_C
            
            min_lambda = 1e-6 * trace_C / covariance_matrix.shape[0]
            return max(lambda_reg, min_lambda)
        except:
            return self.alpha * 100
    
    def predict(self, X):
        """Predict class labels using minimum probability of error rule"""
        X_t = X.T
        n_samples = X.shape[0]
        
        posteriors = np.zeros((n_samples, len(self.classes)))
        
        for i, class_label in enumerate(self.classes):
            mu = self.means[class_label]
            Sigma = self.regularized_covariances[class_label]
            prior = self.priors[class_label]
            
            likelihood = self.eval_gaussian_pdf(X_t, mu, Sigma)
            posteriors[:, i] = likelihood * prior
        
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        
        predictions = self.classes[np.argmax(posteriors, axis=1)]
        
        return predictions