import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve

class OptimalBayesClassifier:
    """
    Optimal Bayes Classifier using true GMM parameters
    Formula: argmax_{L} P(L|x) = argmax_{L} [p(x|L) * P(L)]
    """
    
    def __init__(self, priors, means, covs, weights):
        """
        Initialize with true GMM parameters
        """
        self.priors = priors
        self.means = means
        self.covs = covs
        self.weights = weights
        
        # Validate inputs
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters"""
        assert len(self.priors) == 2, "Priors must have 2 elements"
        assert abs(sum(self.priors) - 1.0) < 1e-10, "Priors must sum to 1"
        assert len(self.means) == 2, "Means must have 2 classes"
        assert len(self.covs) == 2, "Covs must have 2 classes"
        assert len(self.weights) == 2, "Weights must have 2 classes"
    
    def _compute_class_conditional_density(self, x, class_idx):
        """
        Compute class-conditional density p(x|L=class_idx)
        Formula: p(x|L) = Σ_{k} w_{Lk} * N(x|μ_{Lk}, Σ_{Lk})
        """
        density = 0.0
        for comp_idx in range(len(self.weights[class_idx])):
            try:
                component_density = multivariate_normal.pdf(
                    x, 
                    mean=self.means[class_idx][comp_idx], 
                    cov=self.covs[class_idx][comp_idx]
                )
                density += self.weights[class_idx][comp_idx] * component_density
            except Exception as e:
                print(f"Warning: Error computing Gaussian PDF for class {class_idx}, component {comp_idx}: {e}")
                # Fallback: use a small constant to avoid division by zero
                density += 1e-10
        
        return max(density, 1e-10)  # Ensure positive value
    
    def compute_posterior(self, x):
        """
        Compute posterior probability P(L=1|x)
        Formula: P(L=1|x) = [p(x|L=1) * P(L=1)] / [Σ_{L} p(x|L) * P(L)]
        """
        # Ensure x is 1D array
        x = np.asarray(x).flatten()
        
        # Compute class-conditional densities
        p_x_given_0 = self._compute_class_conditional_density(x, 0)
        p_x_given_1 = self._compute_class_conditional_density(x, 1)
        
        # Bayes rule
        evidence = p_x_given_0 * self.priors[0] + p_x_given_1 * self.priors[1]
        
        # Avoid division by zero
        if evidence == 0:
            return 0.5  # Equal probability if no evidence
        
        p_1_given_x = (p_x_given_1 * self.priors[1]) / evidence
        
        return p_1_given_x
    
    def predict_proba(self, X):
        """
        Compute posterior probabilities for all samples
        Returns P(L=1|x) for each sample
        """
        # Handle both 1D and 2D inputs
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        scores = np.array([self.compute_posterior(x) for x in X])
        return scores
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels using minimum probability of error rule
        Formula: Decide L=1 if P(L=1|x) > threshold, else L=0
        """
        scores = self.predict_proba(X)
        predictions = (scores > threshold).astype(int)
        return predictions, scores
    
    def compute_error_rate(self, X, true_labels, threshold=0.5):
        """
        Compute classification error rate
        Formula: Error = (1/N) * Σ [I(prediction ≠ true_label)]
        """
        predictions, _ = self.predict(X, threshold)
        error_rate = np.mean(predictions != true_labels)
        return error_rate
    
    def compute_roc_curve(self, X, true_labels):
        """
        Compute ROC curve using sklearn
        Formula: FPR = FP / (FP + TN), TPR = TP / (TP + FN)
        """
        scores = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(true_labels, scores)
        return fpr, tpr, thresholds

# def test_optimal_classifier():
#     """Simple test function for optimal Bayes classifier"""
#     try:
#         # True parameters from Question 1
#         priors = [0.6, 0.4]
#         means = [
#             [np.array([-0.9, -1.1]), np.array([0.8, 0.75])],
#             [np.array([-1.1, 0.9]), np.array([0.9, -0.75])]
#         ]
#         C = np.array([[0.75, 0], [0, 1.25]])
#         weights = [[0.5, 0.5], [0.5, 0.5]]
        
#         # Create classifier
#         classifier = OptimalBayesClassifier(priors, means, [[C, C], [C, C]], weights)
        
#         # Generate simple test data manually
#         np.random.seed(42)
#         n_samples = 100
        
#         # Generate samples from class 0
#         n_class0 = int(n_samples * priors[0])
#         samples_0 = []
#         for _ in range(n_class0):
#             comp_idx = np.random.choice(2, p=weights[0])
#             sample = np.random.multivariate_normal(
#                 means[0][comp_idx], C
#             )
#             samples_0.append(sample)
        
#         # Generate samples from class 1  
#         n_class1 = n_samples - n_class0
#         samples_1 = []
#         for _ in range(n_class1):
#             comp_idx = np.random.choice(2, p=weights[1])
#             sample = np.random.multivariate_normal(
#                 means[1][comp_idx], C
#             )
#             samples_1.append(sample)
        
#         X_test = np.vstack([samples_0, samples_1])
#         y_test = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
        
#         # Test predictions
#         predictions, scores = classifier.predict(X_test)
#         error_rate = classifier.compute_error_rate(X_test, y_test)
        
#         print("Optimal Bayes Classifier Test:")
#         print(f"Error rate: {error_rate:.4f}")
#         print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
#         print(f"Predictions: {np.unique(predictions, return_counts=True)}")
        
#         return classifier, X_test, y_test
        
#     except Exception as e:
#         print(f"Error in test: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None

# if __name__ == "__main__":
#     classifier, X, y = test_optimal_classifier()