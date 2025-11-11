import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import joblib

class MLPModelSelector:
    """
    MLP Model with Cross-Validation for Model Order Selection
    """
    
    def __init__(self, candidate_hidden_units=None, activation='tanh', max_iter=1000, random_state=42):
        """
        Initialize MLP model selector
        """
        if candidate_hidden_units is None:
            # Reasonable range of hidden units to try
            self.candidate_hidden_units = [5, 10, 15, 20, 25, 30, 40, 50]
        else:
            self.candidate_hidden_units = candidate_hidden_units
            
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_models = {}  # Store best model for each training size
        self.cv_results = {}   # Store cross-validation results
    
    def find_optimal_hidden_units(self, X_train, y_train, n_splits=10):
        """
        Find optimal number of hidden units using k-fold cross-validation
        Objective: Minimize classification error rate
        """
        best_score = -1  # We'll use accuracy (higher is better)
        best_P = self.candidate_hidden_units[0]
        cv_scores = []
        
        print(f"  Cross-validation for {len(X_train)} samples:")
        
        for P in self.candidate_hidden_units:
            mlp = MLPClassifier(
                hidden_layer_sizes=(P,),
                activation=self.activation,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50
            )
            
            # Perform k-fold cross-validation
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(mlp, X_train, y_train, cv=kfold, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            cv_scores.append({
                'P': P,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            })
            
            print(f"    P={P}: Mean Accuracy = {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_P = P
        
        print(f"  Best: P={best_P} with accuracy {best_score:.4f}")
        return best_P, cv_scores
    
    def train_final_model(self, X_train, y_train, P, n_init=10):
        """
        Train final MLP model with multiple random initializations
        Select model with lowest cross-entropy loss (highest likelihood)
        """
        best_model = None
        best_loss = float('inf')
        best_accuracy = 0
        
        print(f"  Training final model with P={P} ({n_init} initializations):")
        
        for i in range(n_init):
            mlp = MLPClassifier(
                hidden_layer_sizes=(P,),
                activation=self.activation,
                max_iter=self.max_iter,
                random_state=self.random_state + i,  # Different seed each time
                early_stopping=False  # Train on full dataset for final model
            )
            
            mlp.fit(X_train, y_train)
            
            # Compute training loss (cross-entropy)
            train_loss = log_loss(y_train, mlp.predict_proba(X_train))
            train_accuracy = accuracy_score(y_train, mlp.predict(X_train))
            
            if train_loss < best_loss:
                best_loss = train_loss
                best_accuracy = train_accuracy
                best_model = mlp
            
            if (i + 1) % 5 == 0:
                print(f"    Init {i+1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")
        
        print(f"  Final: Loss = {best_loss:.4f}, Accuracy = {best_accuracy:.4f}")
        return best_model
    
    def train_all_models(self, datasets):
        """
        Train MLP models for all training set sizes
        """
        train_sizes = [100, 500, 1000, 5000, 10000]
        
        for size in train_sizes:
            print(f"\n=== Training MLP for {size} samples ===")
            X_train, y_train = datasets[f'train_{size}']
            
            # Step 1: Find optimal number of hidden units using cross-validation
            best_P, cv_scores = self.find_optimal_hidden_units(X_train, y_train)
            self.cv_results[size] = cv_scores
            
            # Step 2: Train final model with optimal P and multiple initializations
            best_model = self.train_final_model(X_train, y_train, best_P)
            self.best_models[size] = {
                'model': best_model,
                'P': best_P,
                'training_size': size
            }
        
        return self.best_models
    
    def evaluate_on_test(self, datasets, optimal_classifier):
        """
        Evaluate all trained MLPs on test set and compare with optimal classifier
        """
        X_test, y_test = datasets['test']
        
        print("\n" + "="*60)
        print("FINAL PERFORMANCE EVALUATION")
        print("="*60)
        
        # Get optimal classifier performance
        optimal_error = optimal_classifier.compute_error_rate(X_test, y_test)
        optimal_accuracy = 1 - optimal_error
        
        results = {}
        results['optimal'] = {
            'error_rate': optimal_error,
            'accuracy': optimal_accuracy,
            'P': 'N/A',
            'training_size': 'N/A'
        }
        
        print(f"\nTheoretical Optimal Classifier:")
        print(f"  Test Error Rate: {optimal_error:.4f}")
        print(f"  Test Accuracy:   {optimal_accuracy:.4f}")
        
        # Evaluate each MLP
        for size, model_info in self.best_models.items():
            model = model_info['model']
            P = model_info['P']
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            error_rate = 1 - accuracy
            test_loss = log_loss(y_test, y_proba)
            
            results[size] = {
                'error_rate': error_rate,
                'accuracy': accuracy,
                'test_loss': test_loss,
                'P': P,
                'training_size': size
            }
            
            print(f"\nMLP (trained on {size} samples, P={P}):")
            print(f"  Test Error Rate: {error_rate:.4f}")
            print(f"  Test Accuracy:   {accuracy:.4f}")
            print(f"  Test Loss:       {test_loss:.4f}")
            print(f"  Gap to Optimal:  {error_rate - optimal_error:.4f}")
        
        return results
    
    def plot_results(self, results, save_path='mlp_performance.png'):
        """
        Create the required plot: Test P(error) vs Number of Training Samples
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        training_sizes = []
        mlp_errors = []
        mlp_accuracies = []
        optimal_error = results['optimal']['error_rate']
        optimal_accuracy = results['optimal']['accuracy']
        
        for size in [100, 500, 1000, 5000, 10000]:
            if size in results:
                training_sizes.append(size)
                mlp_errors.append(results[size]['error_rate'])
                mlp_accuracies.append(results[size]['accuracy'])
        
        # Plot 1: Error rate vs training samples (semilog-x)
        ax1.semilogx(training_sizes, mlp_errors, 'bo-', linewidth=2, markersize=8, label='MLP Error Rate')
        ax1.axhline(y=optimal_error, color='r', linestyle='--', linewidth=2, label='Theoretical Optimal Error')
        ax1.set_xlabel('Number of Training Samples')
        ax1.set_ylabel('Test Error Rate')
        ax1.set_title('MLP Performance vs Training Data Size\n(Error Rate)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add annotations for P values
        for i, size in enumerate(training_sizes):
            P = results[size]['P']
            ax1.annotate(f'P={P}', (size, mlp_errors[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Accuracy vs training samples
        ax2.semilogx(training_sizes, mlp_accuracies, 'go-', linewidth=2, markersize=8, label='MLP Accuracy')
        ax2.axhline(y=optimal_accuracy, color='r', linestyle='--', linewidth=2, label='Theoretical Optimal Accuracy')
        ax2.set_xlabel('Number of Training Samples')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('MLP Performance vs Training Data Size\n(Accuracy)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_cv_heatmap(self, save_path='cv_heatmap.png'):
        """
        Create cross-validation performance heatmap
        """
        train_sizes = [100, 500, 1000, 5000, 10000]
        P_values = [5, 10, 15, 20, 25, 30, 40, 50]
        performance_matrix = np.zeros((len(train_sizes), len(P_values)))
        
        for i, size in enumerate(train_sizes):
            if size in self.cv_results:
                for j, P in enumerate(P_values):
                    for cv_result in self.cv_results[size]:
                        if cv_result['P'] == P:
                            performance_matrix[i, j] = cv_result['mean_score']
                            break
        
        plt.figure(figsize=(12, 8))
        im = plt.imshow(performance_matrix, cmap='viridis', aspect='auto')
        plt.xticks(range(len(P_values)), P_values)
        plt.yticks(range(len(train_sizes)), train_sizes)
        plt.xlabel('Number of Hidden Units (P)')
        plt.ylabel('Training Set Size')
        plt.title('Cross-Validation Accuracy Heatmap\n(Brighter = Better Performance)', 
                 fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('CV Accuracy')
        
        # Add text annotations in heatmap
        for i in range(len(train_sizes)):
            for j in range(len(P_values)):
                text = plt.text(j, i, f'{performance_matrix[i, j]:.3f}',
                              ha="center", va="center", color="w" if performance_matrix[i, j] < 0.7 else "k",
                              fontsize=9, fontweight='bold')
        
        # Highlight the best P for each training size
        for i, size in enumerate(train_sizes):
            if size in self.best_models:
                best_P = self.best_models[size]['P']
                j = P_values.index(best_P)
                # Draw a rectangle around the best P
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
                plt.gca().add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_performance_summary_table(results, best_models):
    """
    Create a nice summary table of the results
    """
    train_sizes = [100, 500, 1000, 5000, 10000]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Training Size':<15} {'Optimal P':<10} {'Test Error':<12} {'Optimal Error':<15} {'Gap':<10} {'% of Optimal':<15}")
    print("-"*80)
    
    for size in train_sizes:
        test_error = results[size]['error_rate']
        optimal_error = results['optimal']['error_rate']
        gap = test_error - optimal_error
        percent_optimal = (1 - gap/optimal_error) * 100 if optimal_error > 0 else 100
        
        print(f"{size:<15} {best_models[size]['P']:<10} {test_error:<12.4f} {optimal_error:<15.4f} {gap:<10.4f} {percent_optimal:<15.1f}%")
    
    print("="*80)

# Main execution function
def main():
    """
    Main function 
    """
    # Load data and optimal classifier
    from Theoretical_Optimal_Classifier import load_and_setup
    datasets, optimal_classifier, expected_error = load_and_setup()
    
    print("Starting MLP Model Selection and Training...")
    print(f"Theoretical Optimal Error: {expected_error:.4f}")
    
    # Initialize MLP model selector
    mlp_selector = MLPModelSelector(
        candidate_hidden_units=[5, 10, 15, 20, 25, 30, 40, 50],
        activation='tanh',
        max_iter=2000,
        random_state=42
    )
    
    # Train all MLP models
    best_models = mlp_selector.train_all_models(datasets)
    
    # Evaluate on test set
    results = mlp_selector.evaluate_on_test(datasets, optimal_classifier)
    
    # Create the main performance plot (required by assignment)
    mlp_selector.plot_results(results, 'mlp_performance_comparison.png')
    
    # Create cross-validation heatmap
    mlp_selector.plot_cv_heatmap('cv_performance_heatmap.png')
    
    # Create performance summary table
    create_performance_summary_table(results, mlp_selector.best_models)
    
    # Save results for later analysis
    joblib.dump({
        'best_models': best_models,
        'cv_results': mlp_selector.cv_results,
        'test_results': results
    }, 'mlp_experiment_results.joblib')
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    
    return mlp_selector, results

if __name__ == "__main__":
    mlp_selector, results = main()