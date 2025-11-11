import numpy as np
from scipy.stats import multivariate_normal

class TheoreticalOptimalClassifier:
    """
    Theoretical Optimal Bayes Classifier for 4-class 3D Gaussian problem
    Uses true distribution parameters for MAP classification
    Modelfied from old Optimal_Bayes_Classifier.py
    """
    
    def __init__(self, priors, means, covariances):
        """
        Initialize with true distribution parameters
        """
        self.priors = priors  # [0.25, 0.25, 0.25, 0.25]
        self.means = means    # List of 4 mean vectors (each 3D)
        self.covariances = covariances  # List of 4 covariance matrices (each 3x3)
        self.n_classes = len(priors)
    
    def compute_posterior(self, x):
        """
        Compute posterior probabilities P(ωⱼ | x) for all classes
        Formula: P(ωⱼ | x) ∝ p(x | ωⱼ) * P(ωⱼ)
        """
        posteriors = np.zeros(self.n_classes)
        
        for class_idx in range(self.n_classes):
            # Compute likelihood p(x | ωⱼ)
            likelihood = multivariate_normal.pdf(
                x, 
                mean=self.means[class_idx], 
                cov=self.covariances[class_idx]
            )
            # Compute posterior (unnormalized)
            posteriors[class_idx] = likelihood * self.priors[class_idx]
        
        # Normalize to get proper probabilities
        if np.sum(posteriors) > 0:
            posteriors = posteriors / np.sum(posteriors)
        
        return posteriors
    
    def predict_proba(self, X):
        """
        Compute posterior probabilities for all samples
        Returns matrix of shape (n_samples, 4)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes))
        
        for i in range(n_samples):
            probabilities[i] = self.compute_posterior(X[i])
        
        return probabilities
    
    def predict(self, X):
        """
        Predict class labels using MAP decision rule
        Formula: decide ω_k if k = argmaxⱼ P(ωⱼ | x)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def compute_error_rate(self, X, y_true):
        """
        Compute classification error rate
        """
        y_pred = self.predict(X)
        error_rate = np.mean(y_pred != y_true)
        return error_rate

# Load your generated data
def load_and_setup():
    """
    Load the saved data and set up the optimal classifier
    """
    # Load the saved data
    data = np.load('dataset_info.npz')
    
    # Extract parameters
    priors = data['priors']
    means = data['means']
    covariances = data['covariances']
    optimal_error = data['optimal_error']
    
    # Extract datasets
    datasets = {}
    train_sizes = [100, 500, 1000, 5000, 10000]
    
    for size in train_sizes:
        datasets[f'train_{size}'] = (data[f'X_train_{size}'], data[f'y_train_{size}'])
    
    datasets['test'] = (data['X_test'], data['y_test'])
    
    # Create optimal classifier
    optimal_classifier = TheoreticalOptimalClassifier(priors, means, covariances)
    
    return datasets, optimal_classifier, optimal_error

# Main execution
if __name__ == "__main__":
    # Load data and create classifier
    datasets, optimal_classifier, expected_error = load_and_setup()
    print(expected_error)