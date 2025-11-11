import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.linalg import inv, det
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import pandas as pd

class Question2temp:    
    def __init__(self, n_repetitions=100, k_values=range(1, 11), n_folds=10):
        self.n_repetitions = n_repetitions
        self.k_values = k_values
        self.n_folds = n_folds
        self.datasets = {}
        self.selection_results = {}
        self.selection_rates = {}
    
    # Data Loading
    def load_datasets(self, filename='question2_gmm_datasets.npz'):
        """Load the pre-generated datasets"""
        print("Loading datasets...")
        data = np.load(filename, allow_pickle=True)
        self.true_gmm_params = data['true_gmm_params'].item()
        self.datasets = data['datasets'].item()
        print(f"  Loaded datasets: {list(self.datasets.keys())}")
        return self.datasets
    
    # GMM Implementation
    class Model_Selection_GMM:
        """Model selectionGMM implementation for cross-validation"""
        
        def __init__(self, n_components=3, max_iters=50, reg_weight=1e-6):
            self.n_components = n_components
            self.max_iters = max_iters
            self.reg_weight = reg_weight
            self.weights = None
            self.means = None
            self.covariances = None
        
        def fit(self, X):
            """Simple EM algorithm for GMM"""
            n_samples, n_features = X.shape
            X = X.T  # Convert to (features, samples) format
            
            # Initialize parameters
            self.weights = np.ones(self.n_components) / self.n_components
            self.means = X[:, np.random.choice(n_samples, self.n_components, replace=False)]
            self.covariances = np.array([np.cov(X, rowvar=True) + self.reg_weight * np.eye(n_features) 
                                       for _ in range(self.n_components)])
            
            for iteration in range(self.max_iters):
                # E-step: Compute responsibilities
                responsibilities = np.zeros((self.n_components, n_samples))
                for k in range(self.n_components):
                    responsibilities[k] = self.weights[k] * self._gaussian_pdf(X, self.means[:, k], self.covariances[k])
                
                responsibilities /= np.sum(responsibilities, axis=0, keepdims=True)
                
                # M-step: Update parameters
                Nk = np.sum(responsibilities, axis=1)
                self.weights = Nk / n_samples
                
                for k in range(self.n_components):
                    self.means[:, k] = np.sum(responsibilities[k] * X, axis=1) / Nk[k]
                    X_centered = X - self.means[:, k:k+1]
                    self.covariances[k] = (responsibilities[k] * X_centered) @ X_centered.T / Nk[k]
                    self.covariances[k] += self.reg_weight * np.eye(n_features)
            
            return self
        
        def _gaussian_pdf(self, X, mean, cov):
            """Compute Gaussian PDF"""
            n_features = X.shape[0]
            X_centered = X - mean.reshape(-1, 1)
            try:
                inv_cov = inv(cov)
                det_cov = det(cov)
                constant = 1.0 / (np.power(2 * np.pi, n_features / 2) * np.sqrt(det_cov))
                exponent = -0.5 * np.sum(X_centered * (inv_cov @ X_centered), axis=0)
                return constant * np.exp(exponent)
            except:
                return 1e-10  # Fallback for numerical issues
        
        def score(self, X):
            X = X.T  # (features, samples)
            n_samples = X.shape[1]
            pdf_matrix = np.zeros((self.n_components, n_samples))
            for k in range(self.n_components):
                pdf_matrix[k, :] = self.weights[k] * self._gaussian_pdf(X, self.means[:, k], self.covariances[k])
            log_likelihood = np.sum(np.log(np.sum(pdf_matrix, axis=0) + 1e-10))
            return log_likelihood

    

    # Cross-Validation and Model Selection    
    def cross_validate_gmm(self, X):
        """Perform 10-fold CV to select best number of components"""
        n_samples = X.shape[0]
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        
        k_scores = {}
        
        for k in self.k_values:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                
                try:
                    # Train GMM with k components
                    gmm = self.Model_Selection_GMM(n_components=k, max_iters=50)
                    gmm.fit(X_train)
                    
                    # Compute validation log-likelihood
                    val_score = gmm.score(X_val)
                    fold_scores.append(val_score)
                    
                except Exception as e:
                    # If training fails, use a very low score
                    fold_scores.append(-1e10)
            
            # Average across folds
            k_scores[k] = np.mean(fold_scores)
        
        # Select K with highest average validation log-likelihood
        best_k = max(k_scores, key=k_scores.get)
        return best_k
    
    def run_model_selection_experiment(self):
        """Run complete model selection experiment"""
        print("\nStarting Model Selection Experiment")
        print("=" * 50)
        
        for size_key, size_datasets in self.datasets.items():
            print(f"\nProcessing {size_key} datasets...")
            selection_counts = {k: 0 for k in range(1, 11)}
            
            for i, dataset in enumerate(size_datasets):
                if (i + 1) % 20 == 0:
                    print(f"  Completed {i + 1}/{len(size_datasets)} repetitions")
                
                X = dataset['X']
                best_k = self.cross_validate_gmm(X)
                selection_counts[best_k] += 1
            
            self.selection_results[size_key] = selection_counts
            print(f"  Selection counts for {size_key}: {selection_counts}")
        
        return self.selection_results
    
    # Results Analysis
    def calculate_selection_rates(self):
        """Calculate selection rates from counts"""
        self.selection_rates = {}
        for size_key, counts in self.selection_results.items():
            total = sum(counts.values())
            self.selection_rates[size_key] = {k: counts[k]/total for k in range(1, 11)}
        return self.selection_rates
    
    def print_results_table(self):
        """Print selection rate table"""
        print("\n" + "=" * 60)
        print("GMM MODEL SELECTION RESULTS")
        print("=" * 60)
        print("\nSelection Rates (100 repetitions each):")
        print("K = Number of Gaussian Components\n")
        
        header = "K  " + "".join([f"{size:>10}" for size in ['N=10', 'N=100', 'N=1000']])
        print(header)
        print("-" * len(header))
        
        for k in range(1, 11):
            row = f"{k:2} "
            for size_key in ['N_10', 'N_100', 'N_1000']:
                rate = self.selection_rates[size_key][k]
                row += f"{rate:>10.3f}"
            print(row)
    
    def print_analysis_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        true_k = 4
        for size_key in ['N_10', 'N_100', 'N_1000']:
            rates = self.selection_rates[size_key]
            correct_rate = rates[true_k]
            most_frequent_k = max(rates, key=rates.get)
            underfitting_rate = sum(rates[k] for k in range(1, 4))
            overfitting_rate = sum(rates[k] for k in range(5, 11))
            
            print(f"\n{size_key}:")
            print(f"  Correct selection rate (K=4): {correct_rate:.3f}")
            print(f"  Most frequently selected K: {most_frequent_k}")
            print(f"  Underfitting rate (K<4): {underfitting_rate:.3f}")
            print(f"  Overfitting rate (K>4): {overfitting_rate:.3f}")
            
            if correct_rate > 0.5:
                print("  Model selection works well")
            elif correct_rate > 0.25:
                print("  Model selection is moderate")
            else:
                print("  Model selection performs poorly")
    
    def plot_results(self):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Selection rates for each sample size
        sample_sizes = ['N_10', 'N_100', 'N_1000']
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        for idx, (size_key, color) in enumerate(zip(sample_sizes, colors)):
            rates = self.selection_rates[size_key]
            k_values = list(rates.keys())
            rates_values = [rates[k] for k in k_values]
            
            bars = axes[0,0].bar(np.array(k_values) + idx*0.25 - 0.25, rates_values, 
                                width=0.25, alpha=0.8, color=color, label=size_key)
            
            # Add value labels
            for bar, rate in zip(bars, rates_values):
                if rate > 0.05:
                    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{rate:.2f}', ha='center', va='bottom', fontsize=8)
        
        axes[0,0].set_xlabel('Number of Components (K)')
        axes[0,0].set_ylabel('Selection Rate')
        axes[0,0].set_title('Model Selection Rates by Sample Size')
        axes[0,0].set_xticks(range(1, 11))
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Correct selection rate (K=4) vs sample size
        correct_rates = []
        sample_numbers = [10, 100, 1000]
        
        for size_key in sample_sizes:
            correct_rates.append(self.selection_rates[size_key][4])
        
        axes[0,1].plot(sample_numbers, correct_rates, 'ro-', linewidth=3, markersize=8)
        axes[0,1].set_xlabel('Number of Training Samples')
        axes[0,1].set_ylabel('Correct Selection Rate (K=4)')
        axes[0,1].set_title('Performance Improvement with Sample Size')
        axes[0,1].set_xscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(sample_numbers, correct_rates)):
            axes[0,1].annotate(f'{y:.3f}', (x, y), xytext=(5, 5), 
                             textcoords='offset points', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot 3: Error analysis (underfitting/overfitting)
        underfitting_rates = {}
        overfitting_rates = {}
        correct_rates = {}
        
        for size_key in sample_sizes:
            rates = self.selection_rates[size_key]
            underfitting_rates[size_key] = sum(rates[k] for k in range(1, 4))  # K < 4
            correct_rates[size_key] = rates[4]  # K = 4
            overfitting_rates[size_key] = sum(rates[k] for k in range(5, 11))  # K > 4
        
        x = np.arange(len(sample_sizes))
        width = 0.25
        
        axes[1,0].bar(x - width, [underfitting_rates[size] for size in sample_sizes], 
                     width, label='Underfitting (K<4)', alpha=0.7, color='red')
        axes[1,0].bar(x, [correct_rates[size] for size in sample_sizes], 
                     width, label='Correct (K=4)', alpha=0.7, color='green')
        axes[1,0].bar(x + width, [overfitting_rates[size] for size in sample_sizes], 
                     width, label='Overfitting (K>4)', alpha=0.7, color='blue')
        
        axes[1,0].set_xlabel('Sample Size')
        axes[1,0].set_ylabel('Rate')
        axes[1,0].set_title('Error Analysis: Underfitting vs Overfitting')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(sample_sizes)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of selection rates
        heatmap_data = np.zeros((len(self.k_values), len(sample_sizes)))
        for i, k in enumerate(self.k_values):
            for j, size_key in enumerate(sample_sizes):
                heatmap_data[i, j] = self.selection_rates[size_key][k]
        
        im = axes[1,1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_xlabel('Sample Size')
        axes[1,1].set_ylabel('Number of Components (K)')
        axes[1,1].set_title('Selection Rate Heatmap')
        axes[1,1].set_xticks(range(len(sample_sizes)))
        axes[1,1].set_xticklabels(sample_sizes)
        axes[1,1].set_yticks(range(len(self.k_values)))
        axes[1,1].set_yticklabels(self.k_values)
        
        # Add text annotations in heatmap
        for i in range(len(self.k_values)):
            for j in range(len(sample_sizes)):
                text = axes[1,1].text(j, i, f'{heatmap_data[i, j]:.2f}',
                                    ha="center", va="center", 
                                    color="white" if heatmap_data[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=axes[1,1], label='Selection Rate')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
def main():
    """Main execution function"""
    analyzer = Question2temp(n_repetitions=100)

    analyzer.load_datasets('question2_gmm_datasets.npz')
    analyzer.run_model_selection_experiment()
    analyzer.calculate_selection_rates()
    analyzer.print_results_table()
    analyzer.print_analysis_summary()
    analyzer.plot_results()

    return analyzer, analyzer.selection_results
if __name__ == "__main__":
    analyzer, results = main()