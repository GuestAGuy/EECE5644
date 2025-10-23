import numpy as np
from scipy.stats import multivariate_normal

class GenerateDataFromGMM:
    def generate_data_from_gmm(N, gmmParameters):
        """
        Generates N vector samples from the specified mixture of Gaussians
        Formula: x ~ Σ [prior_i * N(μ_i, Σ_i)]
        
        Parameters:
        -----------
        N : int
            Number of samples to generate
        gmm_parameters : dict
            Dictionary containing:
            - 'priors' : array-like, prior probabilities for each component (should sum to 1)
            - 'meanVectors' : 2D array, mean vectors (n_features x n_components)
            - 'covMatrices' : 3D array, covariance matrices (n_features x n_features x n_components)
        
        Returns:
        --------
        x : ndarray
            Generated data (n_features x N)
        labels : ndarray
            Component labels (1 x N)
        """
        priors = gmmParameters['priors'] # priors should be a row vector
        meanVectors = gmmParameters['meanVectors']
        covMatrices = gmmParameters['covMatrices']
        n = meanVectors.shape[0] # Data dimensionality
        C = len(priors) # Number of components
        x = np.zeros((n,N))
        labels = np.zeros((1,N))
        
        # Decide randomly which samples will come from each component
        u = np.random.random((1,N))
        thresholds = np.zeros((1,C+1))
        thresholds[:,0:C] = np.cumsum(priors)
        thresholds[:,C] = 1
        
        for l in range(C):
            indl = np.where(u <= float(thresholds[:,l]))
            Nl = len(indl[1])
            labels[indl] = (l+1)*1
            u[indl] = 1.1
            x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
        
        return x, labels

    # def test_gmm_data_generator():
        
    #     # Define GMM parameters (same structure as MATLAB)
    #     gmm_parameters = {
    #         'priors': np.array([0.3, 0.5, 0.2]),  # Must sum to 1
    #         'meanVectors': np.array([
    #             [1, 4, 7],    # x-coordinates for 3 components
    #             [2, 5, 8]     # y-coordinates for 3 components
    #         ]),
    #         'covMatrices': np.zeros((2, 2, 3))
    #     }
        
    #     # Define covariance matrices for 3 components
    #     gmm_parameters['covMatrices'][:, :, 0] = np.array([[2, 0.5], [0.5, 1]])
    #     gmm_parameters['covMatrices'][:, :, 1] = np.array([[1, 0.2], [0.2, 2]])
    #     gmm_parameters['covMatrices'][:, :, 2] = np.array([[3, 0.1], [0.1, 1]])
        
    #     print("GMM Parameters:")
    #     print(f"Priors: {gmm_parameters['priors']}")
    #     print(f"Means:\n{gmm_parameters['meanVectors']}")
    #     print(f"Covariances shape: {gmm_parameters['covMatrices'].shape}")
        
    #     # Generate samples
    #     N = 1000
    #     x, labels = generate_data_from_gmm(N, gmm_parameters)
        
    #     print(f"\nGenerated {N} samples:")
    #     print(f"Data shape: {x.shape}")
    #     print(f"Labels shape: {labels.shape}")
        
    #     # Verify distribution matches priors
    #     unique, counts = np.unique(labels, return_counts=True)
    #     empirical_priors = counts / N
    #     print(f"\nEmpirical priors: {empirical_priors}")
    #     print(f"True priors: {gmm_parameters['priors']}")
        
    #     # Plot the results (if 2D)
    #     if x.shape[0] == 2:
    #         import matplotlib.pyplot as plt
            
    #         plt.figure(figsize=(10, 6))
    #         colors = ['red', 'green', 'blue']
            
    #         for i, label in enumerate(unique):
    #             mask = (labels[0] == label)
    #             plt.scatter(x[0, mask], x[1, mask], 
    #                        c=colors[i], alpha=0.6, 
    #                        label=f'Component {int(label)}')
            
    #         # Plot true means
    #         plt.scatter(gmm_parameters['meanVectors'][0, :], 
    #                    gmm_parameters['meanVectors'][1, :], 
    #                    c='black', marker='x', s=200, linewidth=3, 
    #                    label='True Means')
            
    #         plt.xlabel('x1')
    #         plt.ylabel('x2')
    #         plt.title('Generated GMM Data')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.axis('equal')
    #         plt.show()
        
    #     return x, labels

    # # Class-based usage example
    # def test():    
    #     gmm_params = {
    #         'priors': np.array([0.4, 0.6]),
    #         'meanVectors': np.array([
    #             [0, 5],
    #             [0, 5]
    #         ]),
    #         'covMatrices': np.zeros((2, 2, 2))
    #     }
        
    #     gmm_params['covMatrices'][:, :, 0] = np.array([[1, 0], [0, 1]])
    #     gmm_params['covMatrices'][:, :, 1] = np.array([[2, 0.8], [0.8, 2]])
        
    #     generator = generate_data_from_gmm(gmm_params)
    #     x, labels = generator.generate(500)
        
    #     print("Class-based generation:")
    #     print(f"Generated data shape: {x.shape}")
        
    #     return x, labels

# if __name__ == "__main__":
#     # Run the test
#     x, labels = test_gmm_data_generator()
    
#     # Also demonstrate class-based usage
#     x2, labels2 = test()
