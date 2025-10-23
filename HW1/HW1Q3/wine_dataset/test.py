"""
Test the Gaussian classifier implementation
"""

import numpy as np
from gaussian_classifier import GaussianClassifier

def test_classifier():
    print("Testing Gaussian Classifier...")
    
    # Create simple test data
    np.random.seed(42)
    
    # Class 0: centered at [0, 0]
    X0 = np.random.normal(0, 1, (50, 2))
    y0 = np.zeros(50)
    
    # Class 1: centered at [2, 2]  
    X1 = np.random.normal(2, 1, (50, 2))
    y1 = np.ones(50)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    print(f"Test data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    # Train classifier
    clf = GaussianClassifier(alpha=0.01)
    clf.fit(X, y)
    
    # Test predictions
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Error probability: {1-accuracy:.4f}")
    
    # Test probabilities
    proba = clf.predict_proba(X[:5])  # First 5 samples
    print(f"Probability shape: {proba.shape}")
    print("✓ Classifier test passed!")
    
    return clf

if __name__ == "__main__":
    test_classifier()