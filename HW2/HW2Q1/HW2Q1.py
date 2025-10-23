import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Generate_Data_From_GMM import GenerateDataFromGMM
from Optimal_Bayes_Classifier import OptimalBayesClassifier
from Logistic_Regression_Classifier import LogisticRegressionClassifier

class Question1Solution:
    """Complete solution for Question 1: GMM Classification Analysis"""
    
    def __init__(self):
        self.priors = [0.6, 0.4]
        self.means = [
            [np.array([-0.9, -1.1]), np.array([0.8, 0.75])],
            [np.array([-1.1, 0.9]), np.array([0.9, -0.75])]
        ]
        self.C = np.array([[0.75, 0], [0, 1.25]])
        self.weights = [[0.5, 0.5], [0.5, 0.5]]
        self.datasets = {}
        self.results = {'part1': {}, 'part2': {}}
    
    def generate_datasets(self):
        """Generate datasets using true GMM distribution"""
        gmm_params = {
            'priors': np.array([0.3, 0.3, 0.2, 0.2]),
            'meanVectors': np.array([
                [-0.9, 0.8, -1.1, 0.9],
                [-1.1, 0.75, 0.9, -0.75]
            ]),
            'covMatrices': np.zeros((2, 2, 4))
        }
        
        for i in range(4):
            gmm_params['covMatrices'][:, :, i] = self.C
        
        dataset_sizes = {
            'train_50': 50,
            'train_500': 500, 
            'train_5000': 5000,
            'validate_10000': 10000
        }
        
        for name, size in dataset_sizes.items():
            x, comp_labels = GenerateDataFromGMM.generate_data_from_gmm(size, gmm_params)
            class_labels = np.where(comp_labels[0] <= 2, 0, 1)
            self.datasets[name] = (x.T, class_labels)
    
    def run_part1(self):
        """Part 1: Theoretically Optimal Classifier"""
        X_val, y_val = self.datasets['validate_10000']
        
        optimal_classifier = OptimalBayesClassifier(
            priors=self.priors,
            means=self.means,
            covs=[[self.C, self.C], [self.C, self.C]],
            weights=self.weights
        )
        
        predictions, scores = optimal_classifier.predict(X_val)
        error_rate = optimal_classifier.compute_error_rate(X_val, y_val)
        fpr, tpr, thresholds = optimal_classifier.compute_roc_curve(X_val, y_val)
        roc_auc = auc(fpr, tpr)
        
        min_error_idx = np.argmin(np.abs(thresholds - 0.5))
        min_error_fpr = fpr[min_error_idx]
        min_error_tpr = tpr[min_error_idx]
        
        self.results['part1'] = {
            'error_rate': error_rate,
            'scores': scores,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'min_error_point': (min_error_fpr, min_error_tpr)
        }
        
        print(f"Optimal Bayes Error Rate: {error_rate:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        self._plot_roc_curve(fpr, tpr, roc_auc, min_error_fpr, min_error_tpr)
        return self.results['part1']
    
    def run_part2(self):
        """Part 2: Logistic Regression Classifiers"""
        X_val, y_val = self.datasets['validate_10000']
        self.results['part2'] = {}
        
        for feature_type in ['linear', 'quadratic']:
            self.results['part2'][feature_type] = {}
            
            for dataset_size in [50, 500, 5000]:
                dataset_name = f'train_{dataset_size}'
                X_train, y_train = self.datasets[dataset_name]
                
                lr_model = LogisticRegressionClassifier(
                    feature_type=feature_type,
                    regularization=0.01
                )
                lr_model.fit(X_train, y_train, verbose=False)
                
                error_rate = lr_model.get_error_rate(X_val, y_val)
                self.results['part2'][feature_type][dataset_size] = {
                    'error_rate': error_rate
                }
        
        self._print_part2_summary()
        return self.results['part2']
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc, min_error_fpr, min_error_tpr):
        """Plot ROC curve with min-P(error) marker"""
        plt.figure(figsize=(10, 8))
        
        plt.plot(fpr, tpr, color='darkblue', linewidth=2, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', 
                linewidth=1, label='Random Classifier')
        plt.scatter(min_error_fpr, min_error_tpr, color='red', s=100, zorder=5,
                label=f'Min-P(error) Point\n(FPR={min_error_fpr:.3f}, TPR={min_error_tpr:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve - Optimal Bayes Classifier')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_part2_summary(self):
        """Print summary table for Part 2 results"""
        optimal_error = self.results['part1']['error_rate']
        
        print("\nLogistic Regression Performance:")
        print(f"{'Model':<25} {'N=50':<10} {'N=500':<10} {'N=5000':<10}")
        print("-"*55)
        print(f"{'Optimal Bayes':<25} {'-':<10} {'-':<10} {optimal_error:.4f}")
        
        for feature_type in ['linear', 'quadratic']:
            for size in [50, 500, 5000]:
                error = self.results['part2'][feature_type][size]['error_rate']
                if size == 50:
                    print(f"{feature_type.title() + ' Logistic':<25} {error:.4f}    ", end="")
                elif size == 500:
                    print(f"{error:.4f}    ", end="")
                else:
                    print(f"{error:.4f}")

def main():
    """Main function for GMM Classification Analysis"""
    solver = Question1Solution()
    
    print("Question 1: GMM Classification Analysis")
    print("Optimal Bayes vs Logistic Regression")
    
    solver.generate_datasets()
    part1_results = solver.run_part1()
    part2_results = solver.run_part2()
    
    optimal_error = part1_results['error_rate']
    best_lr_error = min(
        part2_results['linear'][5000]['error_rate'],
        part2_results['quadratic'][5000]['error_rate']
    )
    
    print(f"\nFinal Results:")
    print(f"Optimal Bayes Error: {optimal_error:.4f}")
    print(f"Best Logistic Error: {best_lr_error:.4f}")
    print(f"Performance Gap: {abs(optimal_error - best_lr_error):.4f}")

if __name__ == "__main__":
    main()