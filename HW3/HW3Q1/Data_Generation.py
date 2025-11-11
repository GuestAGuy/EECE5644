import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FourClass3DGaussian:
    def __init__(self):
        # Uniform priors for 4 classes
        self.priors = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Define mean vectors for 4 classes in 3D
        self.means = [
            np.array([1.0, 1.0, 1.0]),    # Class 0
            np.array([1.0, 0.5, -1.0]),  # Class 1  
            np.array([-0.5, 1.0, -1.0]),  # Class 2
            np.array([-1.0, -1.0, 1.0])   # Class 3
        ]
        
        # Define covariance matrices - we'll adjust these to get 10-20% error
        # Start with moderate covariance for some overlap
        base_cov = np.array([
            [0.8, 0.2, 0.1],
            [0.2, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        self.covariances = [base_cov] * 4
    
    def adjust_for_target_error(self, target_error_range=(0.10, 0.20), max_iterations=20):
        """
        Adjust covariance matrices to achieve target error rate, because i want to
        """
        print("Adjusting parameters for target error rate...")
        
        for iteration in range(max_iterations):
            # Generate a large test set
            X_test, y_test = self.generate_data(10000)
            
            # Compute theoretical optimal error
            error_rate = self.compute_optimal_error(X_test, y_test)
            print(f"Iteration {iteration+1}: Optimal error rate = {error_rate:.3f}")
            
            if target_error_range[0] <= error_rate <= target_error_range[1]:
                print("Target error rate achieved!")
                break
            elif error_rate < target_error_range[0]:
                # Increase overlap by making covariances larger
                scaling_factor = 1.2
                self.covariances = [cov * scaling_factor for cov in self.covariances]
                print(f"  Increasing covariance scale")
            else:
                # Decrease overlap by making covariances smaller
                scaling_factor = 0.8
                self.covariances = [cov * scaling_factor for cov in self.covariances]
                print(f"  Decreasing covariance scale")
    
    def generate_data(self, n_samples):
        """
        Generate samples from the 4-class Gaussian distribution
        """
        # Determine number of samples per class based on priors
        n_per_class = np.random.multinomial(n_samples, self.priors)
        
        X = []
        y = []
        
        for class_idx in range(4):
            n_class_samples = n_per_class[class_idx]
            if n_class_samples > 0:
                class_samples = multivariate_normal.rvs(
                    mean=self.means[class_idx],
                    cov=self.covariances[class_idx],
                    size=n_class_samples
                )
                X.extend(class_samples)
                y.extend([class_idx] * n_class_samples)
        
        return np.array(X), np.array(y)
    
    def compute_optimal_error(self, X, y_true):
        """
        Compute the theoretical optimal (Bayes) error rate
        """
        n_samples = X.shape[0]
        y_pred = []
        
        for i in range(n_samples):
            posteriors = []
            for class_idx in range(4):
                # Compute P(x|class) * P(class)
                likelihood = multivariate_normal.pdf(
                    X[i], 
                    mean=self.means[class_idx], 
                    cov=self.covariances[class_idx]
                )
                posterior = likelihood * self.priors[class_idx]
                posteriors.append(posterior)
            
            # MAP decision
            predicted_class = np.argmax(posteriors)
            y_pred.append(predicted_class)
        
        error_rate = np.mean(np.array(y_pred) != y_true)
        return error_rate
    
    def plot_data(self, X, y, title="4-Class 3D Gaussian Data", save_path=None):
        """
        Visualize the generated data in 3D
        """
        fig = plt.figure(figsize=(12, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        colors = ['red', 'blue', 'green', 'orange']
        labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        
        for class_idx in range(4):
            mask = (y == class_idx)
            ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                    c=colors[class_idx], label=labels[class_idx], alpha=0.6)
        
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_zlabel('X3')
        ax1.set_title(f'{title} - 3D View')
        ax1.legend()
        
        # 2D projections
        ax2 = fig.add_subplot(122)
        for class_idx in range(4):
            mask = (y == class_idx)
            ax2.scatter(X[mask, 0], X[mask, 1], 
                    c=colors[class_idx], label=labels[class_idx], alpha=0.6)
        
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_title(f'{title} - X1-X2 Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data distribution plot saved to {save_path}")
        
        plt.show()


def generate_all_datasets():
    """
    Generate all training and test datasets as specified in the assignment
    """
    # Initialize the data generator
    data_gen = FourClass3DGaussian()
    
    # Adjust parameters to achieve target error rate
    data_gen.adjust_for_target_error(target_error_range=(0.10, 0.20))
    
    # Generate datasets
    train_sizes = [100, 500, 1000, 5000, 10000]
    test_size = 100000
    
    datasets = {}
    
    # Generate training datasets
    for size in train_sizes:
        X_train, y_train = data_gen.generate_data(size)
        datasets[f'train_{size}'] = (X_train, y_train)
        print(f"Generated training set with {size} samples")
    
    # Generate test dataset
    X_test, y_test = data_gen.generate_data(test_size)
    datasets['test'] = (X_test, y_test)
    print(f"Generated test set with {test_size} samples")
    
    # Compute and display theoretical optimal error
    optimal_error = data_gen.compute_optimal_error(X_test, y_test)
    print(f"\nTheoretical optimal error rate: {optimal_error:.4f} ({optimal_error*100:.2f}%)")
    
    return datasets, data_gen



def verify_data_distribution(datasets, data_gen):
    """
    Verify that our data meets the requirements
    """
    print("=== Data Distribution Verification ===")
    
    # Check priors are uniform
    print(f"Class priors: {data_gen.priors}")
    print(f"Sum of priors: {np.sum(data_gen.priors)}")
    
    # Check training set sizes
    for key, (X, y) in datasets.items():
        if key.startswith('train'):
            print(f"{key}: {X.shape[0]} samples, Class distribution: {np.unique(y, return_counts=True)[1]}")
    
    # Check test set size
    X_test, y_test = datasets['test']
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Verify theoretical optimal error is in target range
    optimal_error = data_gen.compute_optimal_error(X_test, y_test)
    print(f"Theoretical optimal error: {optimal_error:.4f}")
    
    if 0.10 <= optimal_error <= 0.20:
        print(" Optimal error is in target range 10-20%")
    else:
        print(" Optimal error is outside target range")
    
    return optimal_error



def display_data(datasets, data_gen):
    """
    Create visualizations of the data distribution
    """
    # Plot the largest training set and save it
    X_train_large, y_train_large = datasets['train_10000']
    data_gen.plot_data(X_train_large, y_train_large, save_path='data_distribution.png')
    
    # plot a small sample of test data for clarity
    X_test, y_test = datasets['test']
    # Take a random sample of 2000 points for visualization
    indices = np.random.choice(X_test.shape[0], 2000, replace=False)
    data_gen.plot_data(X_test[indices], y_test[indices])



def save_dataset_info(datasets, data_gen, optimal_error, filename='dataset_info.npz'):
    """
    Save the datasets and parameters for later use
    """
    save_dict = {
        'priors': data_gen.priors,
        'means': data_gen.means,
        'covariances': data_gen.covariances,
        'optimal_error': optimal_error
    }
    
    # Add all datasets
    for key, (X, y) in datasets.items():
        save_dict[f'X_{key}'] = X
        save_dict[f'y_{key}'] = y
    
    np.savez(filename, **save_dict)
    print(f"Datasets saved to {filename}")


# Generate all datasets
datasets, data_gen = generate_all_datasets()
# Verify the data
optimal_error = verify_data_distribution(datasets, data_gen)
# Create visualizations
display_data(datasets, data_gen)
# Save everything
save_dataset_info(datasets, data_gen, optimal_error)