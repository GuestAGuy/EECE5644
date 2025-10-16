import numpy as np
import pandas as pd
import os

def load_har_dataset(base_path):
    """Load Human Activity Recognition dataset"""
    print("Loading HAR dataset...")
    
    try:
        # Check if dataset exists
        if not os.path.exists(base_path):
            print("Dataset not found")
            return None, None, None
        
        # Load features
        features_path = os.path.join(base_path, 'features.txt')
        features = pd.read_csv(features_path, sep='\s+', header=None, names=['index', 'feature_name'])
        feature_names = features['feature_name'].tolist()
        
        # Load training data
        X_train = pd.read_csv(os.path.join(base_path, 'train', 'X_train.txt'), sep='\s+', header=None)
        y_train = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'), header=None).squeeze()
        
        # Load test data  
        X_test = pd.read_csv(os.path.join(base_path, 'test', 'X_test.txt'), sep='\s+', header=None)
        y_test = pd.read_csv(os.path.join(base_path, 'test', 'y_test.txt'), header=None).squeeze()
        
        # Combine data
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)
        
        print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X.values, y.values, feature_names
        
    except Exception as e:
        print(f"Load failed: {e}")
        return None, None, None

def main():
    dataset_path = "UCI HAR Dataset" 
    
    X, y, features = load_har_dataset(dataset_path)
    
    if X is None:
        print("FAILED - dataset not loaded")
        return
    
    # TODO: Add Gaussian classifier here once dataset loads
    print("Dataset loaded - need to implement classifier")

if __name__ == "__main__":
    main()