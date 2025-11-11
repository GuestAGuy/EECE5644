# Data_GenerationGMM.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os

class Question2DataGenerator:
    """
    Data Generator for Question 2 - GMM Model Selection with Cross-Validation
    Creates multiple datasets with true 4-component GMM and significant overlap
    """
    
    def __init__(self, n_repetitions=100):
        """
        Initialize data generator for Question 2 requirements
        """
        self.n_repetitions = n_repetitions
        self.true_gmm_params = None
        self.datasets = {}
    
    def define_true_gmm(self):
        """
        Define the true 4-component GMM with significant overlap between two components
        as required by Question 2 specifications
        
        Mathematical Formulation:
        p(x) = Σ_{k=1}^4 π_k * N(x | μ_k, Σ_k)
        where two components overlap significantly (distance ≈ sum of average eigenvalues)
        """
        print("Defining True 4-Component GMM with Overlap")
        print("=" * 50)
        
        # Mixing coefficients - uniform priors for simplicity
        priors = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Mean vectors - two components will be close to create overlap
        mean_vectors = np.array([
            [-0.5, 0.5],  
            [1.5, 1.5],   
            [1.25, -0.25], 
            [-1.0, -1.0]  
        ]).T  # Transpose to get shape (2, 4)
        
        # Covariance matrices 
        cov_matrices = np.zeros((2, 2, 4))
        cov_matrices[:, :, 0] = np.array([[0.8, 0.0],
                                        [0.0, 0.8]])
        cov_matrices[:, :, 1] = np.array([[1.2, 0.5],
                                        [0.5, 0.8]])
        cov_matrices[:, :, 2] = np.array([[0.8, 0.2],
                                        [-0.2, 0.8]])
        cov_matrices[:, :, 3] = np.array([[0.5, -0.3],
                                        [-0.3, 0.7]])
        
        self.true_gmm_params = {
            'priors': priors,
            'meanVectors': mean_vectors,
            'covMatrices': cov_matrices
        }
        
        # Calculate and display overlap metrics
        self._analyze_overlap()
        
        return self.true_gmm_params
    
    def _analyze_overlap(self):
        """Analyze and display overlap characteristics between components"""
        means = self.true_gmm_params['meanVectors']
        covs = self.true_gmm_params['covMatrices']
        
        # Calculate distance between overlapping components (2 and 3)
        mean_2 = means[:, 1]
        mean_3 = means[:, 2]
        distance_23 = np.linalg.norm(mean_2 - mean_3)
        
        # Calculate average eigenvalue sum for covariance matrices
        eigvals_2 = np.linalg.eigvals(covs[:, :, 1])
        eigvals_3 = np.linalg.eigvals(covs[:, :, 2])
        avg_eigenvalue_sum = (np.sum(eigvals_2) + np.sum(eigvals_3)) / 2
        
        print("  True GMM Parameters Defined:")
        print(f"  Priors: {self.true_gmm_params['priors']}")
        print("  Mean Vectors (2D):")
        for i, mean in enumerate(means.T):
            print(f"    Component {i+1}: [{mean[0]:.3f}, {mean[1]:.3f}]")
        
        print(f"\n  Overlap Analysis (Components 2 & 3):")
        print(f"  Distance between means: {distance_23:.3f}")
        print(f"  Sum of eigenvalues (Comp 2): {np.sum(eigvals_2):.3f}")
        print(f"  Sum of eigenvalues (Comp 3): {np.sum(eigvals_3):.3f}")
        print(f"  Average eigenvalue sum: {avg_eigenvalue_sum:.3f}")
        print(f"  Ratio (distance/sum): {distance_23/avg_eigenvalue_sum:.3f}")
        
        # Verify significant overlap condition
        if abs(distance_23 - avg_eigenvalue_sum) / avg_eigenvalue_sum < 0.3:
            print("   2 and 3 have sufficient overlap")
        else:
            print(" Overlap may not be sufficient")
    
    def generate_data_from_gmm(self, N, gmm_parameters):
        """
        Generate N vector samples from the specified mixture of Gaussians
        Formula: x ~ Σ [prior_i * N(μ_i, Σ_i)]
        """
        priors = gmm_parameters['priors']
        mean_vectors = gmm_parameters['meanVectors']
        cov_matrices = gmm_parameters['covMatrices']
        
        n_dims = mean_vectors.shape[0]  # Data dimensionality (2)
        n_components = len(priors)      # Number of components (4)
        
        x = np.zeros((n_dims, N))
        labels = np.zeros((1, N))
        
        # Random component assignments based on priors
        u = np.random.rand(1, N)
        thresholds = np.cumsum(priors)
        thresholds = np.concatenate(([0], thresholds))
        
        for comp_idx in range(n_components):
            # Find indices for current component
            if comp_idx == 0:
                indices = np.where(u <= thresholds[comp_idx+1])[1]
            else:
                indices = np.where((u > thresholds[comp_idx]) & 
                                 (u <= thresholds[comp_idx+1]))[1]
            
            n_samples_comp = len(indices)
            
            if n_samples_comp > 0:
                labels[0, indices] = (comp_idx + 1) * np.ones(n_samples_comp)
                
                # Generate samples from multivariate normal
                samples = np.random.multivariate_normal(
                    mean=mean_vectors[:, comp_idx], 
                    cov=cov_matrices[:, :, comp_idx], 
                    size=n_samples_comp
                ).T
                
                x[:, indices] = samples
        
        return x, labels
    
    def generate_all_datasets(self):
        """
        Generate multiple datasets for cross-validation experiment
        Creates 100 independent datasets for each sample size: 10, 100, 1000
        """
        print("\nStep 2: Generating Multiple Datasets for Cross-Validation")
        print("=" * 50)
        
        dataset_sizes = [10, 100, 1000]
        
        for size in dataset_sizes:
            print(f"\nGenerating {self.n_repetitions} datasets with {size} samples each...")
            size_datasets = []
            
            for rep in range(self.n_repetitions):
                # Generate dataset using our GMM generator
                X, labels = self.generate_data_from_gmm(size, self.true_gmm_params)
                
                # Convert to scikit-learn compatible format
                dataset = {
                    'X': X.T,                    # Shape: (n_samples, 2)
                    'labels': labels[0] - 1,     # Convert to 0-indexed
                    'true_component': labels[0], # Keep original 1-indexed labels
                    'repetition_id': rep,
                    'sample_size': size
                }
                
                size_datasets.append(dataset)
                
                # Print progress
                if (rep + 1) % 20 == 0:
                    print(f"  Completed {rep + 1}/{self.n_repetitions} repetitions")
            
            self.datasets[f'N_{size}'] = size_datasets
            print(f"  Generated {self.n_repetitions} datasets with {size} samples")
            
        
        return self.datasets
    
    def visualize_true_gmm(self, n_samples=1000):
        """
        Visualize the true GMM components to verify overlap
        """        
        # Generate samples for visualization
        X_vis, labels_vis = self.generate_data_from_gmm(n_samples, self.true_gmm_params)
        
        # Extract parameters
        means = self.true_gmm_params['meanVectors']
        covs = self.true_gmm_params['covMatrices']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Generated samples colored by component
        colors = ['red', 'blue', 'green', 'orange']
        component_names = ['Comp 1', 'Comp 2', 'Comp 3', 'Comp 4']
        
        for i in range(4):
            mask = (labels_vis[0] == i + 1)
            axes[0].scatter(X_vis[0, mask], X_vis[1, mask], 
                           c=colors[i], label=component_names[i], 
                           alpha=0.6, s=30)
        
        # Plot true means
        axes[0].scatter(means[0, :], means[1, :], 
                       c='black', marker='x', s=200, linewidth=3,
                       label='True Means')
        
        axes[0].set_xlabel('x₁')
        axes[0].set_ylabel('x₂')
        axes[0].set_title('True GMM: Generated Samples\n')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # Plot 2: True component contours
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        pos = np.dstack((X_grid, Y_grid))
        
        for i in range(4):
            rv = multivariate_normal(means[:, i], covs[:, :, i])
            Z = rv.pdf(pos)
            axes[1].contour(X_grid, Y_grid, Z, levels=3, colors=colors[i], alpha=0.7)
            axes[1].scatter(means[0, i], means[1, i], 
                           c=colors[i], marker='x', s=100, linewidth=2,
                           label=component_names[i])
        
        axes[1].set_xlabel('x₁')
        axes[1].set_ylabel('x₂')
        axes[1].set_title('True GMM: Component Contours\n(Showing Overlap Regions)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        # Plot 3: Verify prior distribution
        unique_labels, counts = np.unique(labels_vis, return_counts=True)
        empirical_priors = counts / n_samples
        true_priors = self.true_gmm_params['priors']
        
        x_pos = np.arange(len(component_names))
        width = 0.35
        
        axes[2].bar(x_pos - width/2, true_priors, width, label='True Priors', alpha=0.7)
        axes[2].bar(x_pos + width/2, empirical_priors, width, label='Empirical Priors', alpha=0.7)
        axes[2].set_xlabel('Component')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Prior Distribution Verification')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(component_names)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_datasets(self, filename='gmm_datasets.npz'):
        """
        Save all generated datasets for later use in cross-validation
        """
        print(f"\n Saving Datasets to '{filename}'")
        
        np.savez(filename,
                 true_gmm_params=self.true_gmm_params,
                 datasets=self.datasets,
                 n_repetitions=self.n_repetitions)
    
    def run_complete_generation(self):
        """
        Run the complete data generation pipeline for Question 2
        """
        print("EECE5644 Assignment 3 - Question 2 Data Generation")
        print("=" * 60)
        
        self.define_true_gmm()
        self.generate_all_datasets()
        self.visualize_true_gmm(n_samples=1000)
        self.save_datasets('question2_gmm_datasets.npz')
        
        return self.true_gmm_params, self.datasets

# Main execution
if __name__ == "__main__":
    # Initialize and run the data generator
    generator = Question2DataGenerator(n_repetitions=100)
    true_params, all_datasets = generator.run_complete_generation()