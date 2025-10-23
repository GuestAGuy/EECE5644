import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function

class LogisticRegressionClassifier:
    """
    Logistic Regression with linear and quadratic features
    Formula: P(L=1|x) = σ(w^T * φ(x)) where σ(z) = 1 / (1 + exp(-z))
    """
    
    def __init__(self, feature_type='linear', regularization=0.01):
        """
        Initialize logistic regression model
        """
        self.feature_type = feature_type
        self.regularization = regularization
        self.weights = None
    
    def transform_features(self, X):
        """
        Transform input features according to model type
        
        Linear features: φ(x) = [1, x1, x2]^T
        Quadratic features: φ(x) = [1, x1, x2, x1², x1x2, x2²]^T
        """
        if self.feature_type == 'linear':
            # Formula: z = [1, x1, x2]^T
            return np.column_stack([np.ones(X.shape[0]), X])
        elif self.feature_type == 'quadratic':
            # Formula: z = [1, x1, x2, x1², x1x2, x2²]^T
            x1, x2 = X[:, 0], X[:, 1]
            return np.column_stack([
                np.ones(X.shape[0]),
                x1, x2,
                x1**2, x1*x2, x2**2
            ])
        else:
            raise ValueError("feature_type must be 'linear' or 'quadratic'")
    
    def negative_log_likelihood(self, weights, X, y):
        """
        Compute negative log-likelihood with L2 regularization
        Formula: J(w) = -Σ [y_i * log(σ(w^Tφ(x_i))) + (1-y_i)*log(1-σ(w^Tφ(x_i)))] + (λ/2)||w||²
        """
        X_feat = self.transform_features(X)
        z = X_feat @ weights
        predictions = expit(z)  # σ(z) = 1/(1+exp(-z))
        
        # Avoid log(0) issues
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Negative log-likelihood
        # Formula: -Σ [y_i * log(h_i) + (1-y_i)*log(1-h_i)]
        nll = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # L2 regularization (excluding bias term)
        # Formula: (λ/2N) * Σ w_j² (for j>=1)
        reg_term = (self.regularization / (2 * len(y))) * np.sum(weights[1:]**2)
        
        return nll + reg_term
    
    def gradient(self, weights, X, y):
        """
        Compute gradient of negative log-likelihood
        Formula: ∇J(w) = (1/N) * Σ (h_i - y_i) * φ(x_i) + (λ/N) * w (for j>=1)
        """
        X_feat = self.transform_features(X)
        z = X_feat @ weights
        predictions = expit(z)
        error = predictions - y
        
        # Gradient without regularization
        # Formula: (1/N) * Σ (h_i - y_i) * φ(x_i)
        gradient = X_feat.T @ error / len(y)
        
        # Add L2 regularization (excluding bias term)
        # Formula: (λ/N) * w_j for j >= 1
        if self.regularization > 0:
            gradient[1:] += (self.regularization / len(y)) * weights[1:]
        
        return gradient
    
    def fit(self, X, y, method='BFGS', verbose=False):
        """
        Fit logistic regression model using optimization
    
        """
        # Initialize weights
        X_feat = self.transform_features(X)
        weights_init = np.zeros(X_feat.shape[1])
        
        # Optimize using scipy
        result = minimize(
            fun=self.negative_log_likelihood,
            x0=weights_init,
            args=(X, y),
            method=method,
            jac=self.gradient,
            options={'maxiter': 1000, 'disp': verbose}
        )
        
        if result.success:
            self.weights = result.x
            if verbose:
                print(f"Optimization successful. Final NLL: {result.fun:.4f}")
        else:
            print(f"Optimization warning: {result.message}")
            self.weights = result.x
        
        return self.weights
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        Formula: P(L=1|x) = σ(w^T * φ(x))
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_feat = self.transform_features(X)
        z = X_feat @ self.weights
        return expit(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        Formula: ŷ = 1 if P(L=1|x) ≥ threshold, else 0
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y_true):
        """
        Compute classification accuracy
        Formula: Accuracy = (1/N) * Σ I(ŷ_i = y_i)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
    
    def get_error_rate(self, X, y_true):
        """
        Compute classification error rate
        Formula: Error = 1 - Accuracy
        """
        return 1 - self.score(X, y_true)

# def test_logistic_regression():
#     """Simple test function for logistic regression"""
#     # Generate simple 2D test data
#     np.random.seed(42)
    
#     # Class 0: centered at (-1, -1)
#     X0 = np.random.multivariate_normal([-1, -1], [[1, 0.5], [0.5, 1]], 50)
#     y0 = np.zeros(50)
    
#     # Class 1: centered at (1, 1)  
#     X1 = np.random.multivariate_normal([1, 1], [[1, -0.5], [-0.5, 1]], 50)
#     y1 = np.ones(50)
    
#     X_train = np.vstack([X0, X1])
#     y_train = np.hstack([y0, y1])
    
#     print("Logistic Regression Test:")
    
#     # Test linear model
#     model_linear = LogisticRegressionClassifier(feature_type='linear')
#     model_linear.fit(X_train, y_train, verbose=True)
#     linear_accuracy = model_linear.score(X_train, y_train)
#     print(f"Linear model accuracy: {linear_accuracy:.4f}")
    
#     # Test quadratic model
#     model_quadratic = LogisticRegressionClassifier(feature_type='quadratic')
#     model_quadratic.fit(X_train, y_train, verbose=True)
#     quadratic_accuracy = model_quadratic.score(X_train, y_train)
#     print(f"Quadratic model accuracy: {quadratic_accuracy:.4f}")
    
#     return model_linear, model_quadratic, X_train, y_train

# if __name__ == "__main__":
#     test_logistic_regression()