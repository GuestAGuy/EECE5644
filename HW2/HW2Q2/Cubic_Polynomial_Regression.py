import numpy as np
from scipy.linalg import inv

class CubicPolynomialRegression:
    """
    Cubic Polynomial Regression with ML and MAP estimation
    Model: y = c(x,w) + v, where v ~ N(0, σ²)
    """
    
    def __init__(self, sigma2=1.0):
        """
        Initialize cubic polynomial regression
        """
        self.sigma2 = sigma2
        self.w = None
        self.feature_names = [
            '1', 'x1', 'x2', 
            'x1²', 'x1x2', 'x2²', 
            'x1³', 'x1²x2', 'x1x2²', 'x2³'
        ]
    
    def cubic_features(self, X):
        """
        Transform 2D input to cubic polynomial features
        Returns design matrix Φ with shape (N, 10)
        
        Features: [1, x1, x2, x1², x1x2, x2², x1³, x1²x2, x1x2², x2³]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        x1, x2 = X[:, 0], X[:, 1]
        
        Phi = np.column_stack([
            np.ones(len(X)),      # 1 (bias)
            x1, x2,              # Linear terms
            x1**2, x1*x2, x2**2, # Quadratic terms  
            x1**3, x1**2*x2, x1*x2**2, x2**3  # Cubic terms
        ])
        
        return Phi
    
    def ml_estimate(self, X, y):
        """
        Maximum Likelihood estimation
        Formula: w_ML = (ΦᵀΦ)⁻¹Φᵀy
        """
        Phi = self.cubic_features(X)
        
        try:
            # Closed-form solution using normal equations
            self.w = inv(Phi.T @ Phi) @ Phi.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.w = np.linalg.pinv(Phi) @ y
        
        return self.w
    
    def map_estimate(self, X, y, gamma):
        """
        Maximum A Posteriori estimation with Gaussian prior
        Formula: w_MAP = (ΦᵀΦ + (σ²/γ)I)⁻¹Φᵀy
        """
        Phi = self.cubic_features(X)
        n_features = Phi.shape[1]
        I = np.eye(n_features)
        
        # MAP solution with regularization
        try:
            self.w = inv(Phi.T @ Phi + (self.sigma2/gamma) * I) @ Phi.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if numerical issues
            self.w = np.linalg.pinv(Phi.T @ Phi + (self.sigma2/gamma) * I) @ Phi.T @ y
        
        return self.w
    
    def predict(self, X):
        """
        Make predictions using estimated weights
        Formula: ŷ = Φw
        """
        if self.w is None:
            raise ValueError("Model not trained. Call ml_estimate() or map_estimate() first.")
        
        Phi = self.cubic_features(X)
        return Phi @ self.w
    
    def get_weights(self):
        """Return estimated weights"""
        return self.w.copy() if self.w is not None else None

def average_squared_error(y_true, y_pred):
    """
    Compute Average Squared Error (ASE)
    Formula: ASE = (1/N) * Σ(y_true - y_pred)²
    """
    return np.mean((y_true - y_pred)**2)