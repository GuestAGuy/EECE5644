# **EECE5644 Assignment 4 - Problem 1: Final Clean Version**

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
import os
import warnings

# Suppress ALL annoying TensorFlow and Keras warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel('ERROR')  
warnings.filterwarnings('ignore')  

# Set matplotlib backend
plt.switch_backend('Agg')

def generate_data(n_samples_train=1000, n_samples_test=10000, r_neg=2, r_pos=4, sigma=1):
    """Generate concentric circle data as described in the problem"""    
    # Training data
    X_train, y_train = [], []
    for label, r in [(-1, r_neg), (1, r_pos)]:
        theta = np.random.uniform(-np.pi, np.pi, n_samples_train//2)
        x = r * np.cos(theta) + np.random.normal(0, sigma, n_samples_train//2)
        y = r * np.sin(theta) + np.random.normal(0, sigma, n_samples_train//2)
        X_train.extend(np.column_stack([x, y]))
        y_train.extend([label] * (n_samples_train//2))
    
    # Test data  
    X_test, y_test = [], []
    for label, r in [(-1, r_neg), (1, r_pos)]:
        theta = np.random.uniform(-np.pi, np.pi, n_samples_test//2)
        x = r * np.cos(theta) + np.random.normal(0, sigma, n_samples_test//2)
        y = r * np.sin(theta) + np.random.normal(0, sigma, n_samples_test//2)
        X_test.extend(np.column_stack([x, y]))
        y_test.extend([label] * (n_samples_test//2))
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def quadratic_activation(x):
    """Custom quadratic activation function: f(x) = x²"""
    return tf.square(x)

def create_quadratic_mlp(hidden_units=50, learning_rate=0.001):
    """Create MLP with quadratic activation in hidden layer"""
    model = Sequential([
        Input(shape=(2,)),  # Proper Input layer
        Dense(hidden_units, activation=quadratic_activation),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def mlp_quadratic_with_cv(X_train, y_train):
    """MLP with quadratic activation and hyperparameter tuning"""
    
    # Convert labels from [-1, 1] to [0, 1] for softmax
    y_train_01 = (y_train == 1).astype(int)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid
    param_grid = {
        'model__hidden_units': [10, 20, 50],
        'model__learning_rate': [0.001, 0.01]
    }
    
    # Create Keras classifier wrapper
    mlp = KerasClassifier(model=create_quadratic_mlp, 
                         epochs=100, 
                         batch_size=32, 
                         verbose=0)
    
    # Perform grid search
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train_scaled, y_train_01)
    
    print("Quadratic MLP Best parameters:", grid_search.best_params_)
    print("Quadratic MLP Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_, grid_search, scaler

def svm_with_cv(X_train, y_train):
    """SVM with RBF kernel and hyperparameter tuning using GridSearchCV"""
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    
    # Perform grid search with 5-fold cross-validation
    svm = SVC(kernel='rbf', random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("SVM Best parameters:", grid_search.best_params_)
    print("SVM Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_, grid_search

def mlp_standard_with_cv(X_train, y_train):
    """Standard MLP with tanh/relu activation and hyperparameter tuning"""
    
    # Scale data for better MLP performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (50,)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001]
    }
    
    # Perform grid search
    mlp = MLPClassifier(max_iter=2000, random_state=42, early_stopping=False, tol=1e-4)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print("Standard MLP Best parameters:", grid_search.best_params_)
    print("Standard MLP Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_, grid_search, scaler

def plot_cv_results(svm_grid, mlp_standard_grid, mlp_quadratic_grid=None):
    """Plot cross-validation results for hyperparameter tuning"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # SVM results
    svm_results = svm_grid.cv_results_
    C_len = len(svm_grid.param_grid['C'])
    gamma_len = len(svm_grid.param_grid['gamma'])
    
    scores = svm_results['mean_test_score'].reshape(C_len, gamma_len)
    
    im = axes[0].imshow(scores, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(gamma_len))
    axes[0].set_yticks(range(C_len))
    axes[0].set_xticklabels([f'{g:.3f}' for g in svm_grid.param_grid['gamma']])
    axes[0].set_yticklabels(svm_grid.param_grid['C'])
    axes[0].set_xlabel('Gamma')
    axes[0].set_ylabel('C')
    axes[0].set_title('SVM Cross-Validation Accuracy')
    plt.colorbar(im, ax=axes[0])
    
    # Add accuracy values to heatmap
    for i in range(C_len):
        for j in range(gamma_len):
            axes[0].text(j, i, f'{scores[i, j]:.3f}', 
                        ha='center', va='center', color='white', fontsize=8)
    
    # Standard MLP results
    mlp_results = mlp_standard_grid.cv_results_
    mlp_df = pd.DataFrame({
        'hidden_units': [str(params['hidden_layer_sizes'][0]) for params in mlp_results['params']],
        'activation': [params['activation'] for params in mlp_results['params']],
        'alpha': [params['alpha'] for params in mlp_results['params']],
        'score': mlp_results['mean_test_score']
    })
    
    # Plot by hidden units, grouped by activation
    activations = mlp_df['activation'].unique()
    colors = ['blue', 'red']
    
    for i, activation in enumerate(activations):
        activation_data = mlp_df[mlp_df['activation'] == activation]
        best_by_size = activation_data.groupby('hidden_units')['score'].max()
        axes[1].plot(best_by_size.index, best_by_size.values, 'o-', 
                    color=colors[i], linewidth=2, markersize=8, label=f'{activation} activation')
        
        for size, score in zip(best_by_size.index, best_by_size.values):
            axes[1].annotate(f'{score:.3f}', (size, score), 
                           textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    
    axes[1].set_xlabel('Number of Hidden Units')
    axes[1].set_ylabel('Cross-Validation Accuracy')
    axes[1].set_title('Standard MLP Cross-Validation Performance')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Quadratic MLP results
    if mlp_quadratic_grid:
        quad_results = mlp_quadratic_grid.cv_results_
        quad_df = pd.DataFrame({
            'hidden_units': [str(params['model__hidden_units']) for params in quad_results['params']],
            'learning_rate': [params['model__learning_rate'] for params in quad_results['params']],
            'score': quad_results['mean_test_score']
        })
        
        learning_rates = quad_df['learning_rate'].unique()
        colors_quad = ['green', 'orange']
        
        for i, lr in enumerate(learning_rates):
            lr_data = quad_df[quad_df['learning_rate'] == lr]
            best_by_size = lr_data.groupby('hidden_units')['score'].max()
            axes[2].plot(best_by_size.index, best_by_size.values, 's-', 
                        color=colors_quad[i], linewidth=2, markersize=8, label=f'lr={lr}')
            
            for size, score in zip(best_by_size.index, best_by_size.values):
                axes[2].annotate(f'{score:.3f}', (size, score), 
                               textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
        
        axes[2].set_xlabel('Number of Hidden Units')
        axes[2].set_ylabel('Cross-Validation Accuracy')
        axes[2].set_title('Quadratic MLP Cross-Validation Performance')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    print("Cross-validation results saved as 'cross_validation_results.png'")

def get_quadratic_mlp_predictions(model, X_data):
    """Helper function to handle quadratic MLP predictions"""
    predictions = model.predict(X_data)
    
    if predictions.ndim == 1:
        return predictions * 2 - 1, predictions
    else:
        class_indices = np.argmax(predictions, axis=1)
        return class_indices * 2 - 1, class_indices

def evaluate_and_visualize(models, model_names, X_test, y_test, scalers=None):
    """Evaluate models and create visualizations"""
    
    if scalers is None:
        scalers = [None] * len(models)
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
    
    for i, (model, name, scaler) in enumerate(zip(models, model_names, scalers)):
        if 'MLP' in name and scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            if 'Quadratic' in name:
                y_pred_original, _ = get_quadratic_mlp_predictions(model, X_test_scaled)
                error_rate = 1 - accuracy_score(y_test, y_pred_original)
            else:
                y_pred = model.predict(X_test_scaled)
                error_rate = 1 - accuracy_score(y_test, y_pred)
        else:
            y_pred = model.predict(X_test)
            error_rate = 1 - accuracy_score(y_test, y_pred)
        
        print(f"{name} Test Error Rate: {error_rate:.4f}")
        
        # Plot decision boundaries
        xx, yy = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
        if 'MLP' in name and scaler is not None:
            grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
            if 'Quadratic' in name:
                grid_predictions_original, _ = get_quadratic_mlp_predictions(model, grid_scaled)
                Z = grid_predictions_original.reshape(xx.shape)
            else:
                Z = model.predict(grid_scaled).reshape(xx.shape)
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        axes[0,i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        axes[0,i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, alpha=0.6, s=1)
        axes[0,i].set_title(f'{name} Decision Boundary\nTest Error: {error_rate:.3f}')
        axes[0,i].set_xlabel('X1')
        axes[0,i].set_ylabel('X2')
        axes[0,i].set_xlim(-6, 6)
        axes[0,i].set_ylim(-6, 6)
        
        # Confusion matrix
        if 'Quadratic' in name:
            cm = confusion_matrix(y_test, y_pred_original)
        elif 'MLP' in name and scaler is not None:
            cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        else:
            cm = confusion_matrix(y_test, model.predict(X_test))
            
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,i], cmap='Blues',
                   xticklabels=['Class -1', 'Class +1'], 
                   yticklabels=['Class -1', 'Class +1'])
        axes[1,i].set_title(f'{name} Confusion Matrix')
        axes[1,i].set_xlabel('Predicted')
        axes[1,i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('test_performance_results.png', dpi=300, bbox_inches='tight')
    print("Test performance results saved as 'test_performance_results.png'")

def main():
    print("=== EECE5644 Assignment 4 - Problem 1 ===")
    print("Comparing SVM, Standard MLP, and Quadratic MLP")
    
    # Generate data
    print("Generating data...")
    X_train, y_train, X_test, y_test = generate_data()
    
    # Plot training data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
                c='red', alpha=0.6, label='Class -1 (r=2)', s=10)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                c='blue', alpha=0.6, label='Class +1 (r=4)', s=10)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Training Data - Concentric Circles')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_data.png', dpi=300, bbox_inches='tight')
    print("Training data visualization saved as 'training_data.png'")
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class distribution: {-1}: {np.sum(y_train == -1)}, +1: {np.sum(y_train == 1)}")
    
    # Train models
    print("\n=== Training SVM with RBF Kernel ===")
    best_svm, svm_grid_search = svm_with_cv(X_train, y_train)
    
    print("\n=== Training Standard MLP ===")
    best_mlp_standard, mlp_standard_grid_search, scaler_standard = mlp_standard_with_cv(X_train, y_train)
    
    print("\n=== Training Quadratic MLP ===")
    best_mlp_quadratic, mlp_quadratic_grid_search, scaler_quadratic = mlp_quadratic_with_cv(X_train, y_train)
    
    # Show results
    print("\n=== Cross-Validation Results ===")
    plot_cv_results(svm_grid_search, mlp_standard_grid_search, mlp_quadratic_grid_search)
    
    print("\n=== Final Evaluation on Test Set ===")
    models = [best_svm, best_mlp_standard, best_mlp_quadratic]
    model_names = ['SVM-RBF', 'Standard MLP', 'Quadratic MLP']
    scalers = [None, scaler_standard, scaler_quadratic]
    evaluate_and_visualize(models, model_names, X_test, y_test, scalers)
    
    # Final summary
    print("\n=== Comprehensive Summary ===")
    svm_test_acc = accuracy_score(y_test, best_svm.predict(X_test))
    X_test_standard_scaled = scaler_standard.transform(X_test)
    mlp_standard_test_acc = accuracy_score(y_test, best_mlp_standard.predict(X_test_standard_scaled))
    X_test_quadratic_scaled = scaler_quadratic.transform(X_test)
    y_pred_quadratic_original, _ = get_quadratic_mlp_predictions(best_mlp_quadratic, X_test_quadratic_scaled)
    mlp_quadratic_test_acc = accuracy_score(y_test, y_pred_quadratic_original)
    
    print(f"SVM Test Accuracy: {svm_test_acc:.4f} (Error: {1-svm_test_acc:.4f})")
    print(f"Standard MLP Test Accuracy: {mlp_standard_test_acc:.4f} (Error: {1-mlp_standard_test_acc:.4f})")
    print(f"Quadratic MLP Test Accuracy: {mlp_quadratic_test_acc:.4f} (Error: {1-mlp_quadratic_test_acc:.4f})")
    
    accuracies = [svm_test_acc, mlp_standard_test_acc, mlp_quadratic_test_acc]
    best_idx = np.argmax(accuracies)
    best_model_name = model_names[best_idx]
    best_accuracy = accuracies[best_idx]
    
    print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")

if __name__ == "__main__":
    main()