import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv, det
from scipy.spatial.distance import cdist
import logging

class GaussianMixtureModel:
    """EM-based Gaussian Mixture Model implementation based on provided EMforGMM.m and EMForGMMv.m MATLAB code reference"""
    
    def __init__(self, n_components=3, delta=1e-2, reg_weight=1e-10, max_iters=100):
        self.n_components = n_components
        self.delta = delta  # EM stopping criterion tolerance
        self.reg_weight = reg_weight  # regularization for covariance
        self.max_iters = max_iters
        self.alpha = None  # mixing coefficients
        self.mu = None     # means
        self.Sigma = None  # covariance matrices
        self.log_likelihood_history = []
        
    def rand_gaussian(self, n, mu, Sigma):
        """Generate samples from Gaussian distribution - same as MATLAB randGaussian function"""
        d = len(mu)
        z = np.random.randn(d, n)
        A = sqrtm(Sigma)
        x = A @ z + mu.reshape(-1, 1)
        return x
    
    def rand_gmm(self, n, alpha, mu, Sigma):
        """Generate samples from GMM - same as MATLAB randGMM function"""
        d = mu.shape[0]
        cum_alpha = np.cumsum(np.concatenate(([0], alpha)))
        u = np.random.rand(n)
        x = np.zeros((d, n))
        
        for m in range(len(alpha)):
            # Find indices for this component
            if m == len(alpha) - 1:
                ind = np.where((cum_alpha[m] <= u) & (u <= cum_alpha[m+1]))[0]
            else:
                ind = np.where((cum_alpha[m] <= u) & (u < cum_alpha[m+1]))[0]
            
            if len(ind) > 0:
                x[:, ind] = self.rand_gaussian(len(ind), mu[:, m], Sigma[:, :, m])
        
        return x
    
    def eval_gaussian(self, x, mu, Sigma):
        """Evaluate Gaussian PDF - same as MATLAB evalGaussian function"""
        n, N = x.shape
        try:
            inv_Sigma = inv(Sigma)
            C = (2 * np.pi) ** (-n / 2) * det(inv_Sigma) ** 0.5
            x_centered = x - mu.reshape(-1, 1)
            E = -0.5 * np.sum(x_centered * (inv_Sigma @ x_centered), axis=0)
            g = C * np.exp(E)
            return g
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            logging.warning("Singular covariance matrix detected, using regularization")
            Sigma_reg = Sigma + self.reg_weight * np.eye(n)
            return self.eval_gaussian(x, mu, Sigma_reg)
    
    def eval_gmm(self, x, alpha, mu, Sigma):
        """Evaluate GMM PDF - same as MATLAB evalGMM"""
        gmm = np.zeros(x.shape[1])
        for m in range(len(alpha)):
            gmm += alpha[m] * self.eval_gaussian(x, mu[:, m], Sigma[:, :, m])
        return gmm
    
    def fit(self, x, initial_params=None):
        """
        EM algorithm for GMM
        
        Parameters:
        x: data matrix (d x N)
        initial_params: optional initial parameters (alpha, mu, Sigma)
        """
        d, N = x.shape
        M = self.n_components
        
        # Initialize parameters
        if initial_params is None:
            # Random initialization
            self.alpha = np.ones(M) / M
            shuffled_indices = np.random.permutation(N)
            self.mu = x[:, shuffled_indices[:M]]
            
            # Assign samples to nearest mean for initial covariance
            distances = cdist(self.mu.T, x.T)
            assigned_labels = np.argmin(distances, axis=0)
            
            self.Sigma = np.zeros((d, d, M))
            for m in range(M):
                cluster_points = x[:, assigned_labels == m]
                if cluster_points.shape[1] > 1:
                    cov_matrix = np.cov(cluster_points, ddof=0)
                else:
                    cov_matrix = np.eye(d)
                self.Sigma[:, :, m] = cov_matrix + self.reg_weight * np.eye(d)
        else:
            self.alpha, self.mu, self.Sigma = initial_params
        
        # EM algorithm
        converged = False
        t = 0
        self.log_likelihood_history = []
        
        while not converged and t < self.max_iters:
            # E-step: Compute responsibilities
            temp = np.zeros((M, N))
            for l in range(M):
                temp[l, :] = self.alpha[l] * self.eval_gaussian(x, self.mu[:, l], self.Sigma[:, :, l])
            
            plgivenx = temp / np.sum(temp, axis=0, keepdims=True)
            
            # M-step: Update parameters
            alpha_new = np.mean(plgivenx, axis=1)
            w = plgivenx / np.sum(plgivenx, axis=1, keepdims=True)
            mu_new = x @ w.T
            
            Sigma_new = np.zeros((d, d, M))
            for l in range(M):
                v = x - mu_new[:, l].reshape(-1, 1)
                u = w[l, :].reshape(1, -1) * v
                Sigma_new[:, :, l] = u @ v.T + self.reg_weight * np.eye(d)
            
            # Check convergence
            D_alpha = np.sum(np.abs(alpha_new - self.alpha))
            D_mu = np.sum(np.abs(mu_new - self.mu))
            D_Sigma = np.sum(np.abs(Sigma_new - self.Sigma))
            
            converged = (D_alpha + D_mu + D_Sigma) < self.delta
            
            # Update parameters
            self.alpha = alpha_new
            self.mu = mu_new
            self.Sigma = Sigma_new
            
            # Compute log-likelihood
            log_likelihood = np.sum(np.log(self.eval_gmm(x, self.alpha, self.mu, self.Sigma)))
            self.log_likelihood_history.append(log_likelihood)
            
            t += 1
            
            # Display progress (optional)
            if d == 2 and t % 10 == 0:  # Only plot every 10 iterations for performance
                self.display_progress(t, x, self.alpha, self.mu, self.Sigma)
        
        print(f"EM algorithm converged after {t} iterations")
        return self
    
    def display_progress(self, t, x, alpha, mu, Sigma):
        """Display progress similar to MATLAB displayProgress function"""
        plt.figure(figsize=(12, 5))
        
        # Plot data and contours
        plt.subplot(1, 2, 1)
        plt.scatter(x[0, :], x[1, :], alpha=0.6, s=1)
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.title(f'Data and Estimated GMM Contours (Iteration {t})')
        
        # Plot GMM contours
        range_x1 = [np.min(x[0, :]), np.max(x[0, :])]
        range_x2 = [np.min(x[1, :]), np.max(x[1, :])]
        
        x1_grid = np.linspace(range_x1[0], range_x1[1], 101)
        x2_grid = np.linspace(range_x2[0], range_x2[1], 91)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        grid_points = np.vstack([X1.ravel(), X2.ravel()])
        z_gmm = self.eval_gmm(grid_points, alpha, mu, Sigma).reshape(X1.shape)
        
        plt.contour(X1, X2, z_gmm, levels=10)
        plt.axis('equal')
        
        # Plot log-likelihood history
        plt.subplot(1, 2, 2)
        plt.plot(range(1, t+1), self.log_likelihood_history[:t], 'b.-')
        plt.xlabel('Iteration Index')
        plt.ylabel('Log-Likelihood of Data')
        plt.title('EM Convergence')
        plt.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def predict_proba(self, x):
        """Predict component responsibilities for new data"""
        M = self.n_components
        N = x.shape[1]
        
        temp = np.zeros((M, N))
        for l in range(M):
            temp[l, :] = self.alpha[l] * self.eval_gaussian(x, self.mu[:, l], self.Sigma[:, :, l])
        
        responsibilities = temp / np.sum(temp, axis=0, keepdims=True)
        return responsibilities.T  # Return as (N x M)
    
    def predict(self, x):
        """Predict component labels for new data"""
        responsibilities = self.predict_proba(x)
        return np.argmax(responsibilities, axis=1)
    
    def score(self, x):
        """Compute log-likelihood of data"""
        return np.sum(np.log(self.eval_gmm(x, self.alpha, self.mu, self.Sigma)))


def test():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # True GMM parameters 
    alpha_true = np.array([0.2, 0.3, 0.5])
    mu_true = np.array([[-10, 0, 10], [0, 0, 0]])
    Sigma_true = np.zeros((2, 2, 3))
    Sigma_true[:, :, 0] = np.array([[3, 1], [1, 20]])
    Sigma_true[:, :, 1] = np.array([[7, 1], [1, 2]])
    Sigma_true[:, :, 2] = np.array([[4, 1], [1, 16]])
    
    # Create GMM instance
    gmm = GaussianMixtureModel(n_components=3, delta=1e-2, reg_weight=1e-10)
    
    # Generate samples (N=1000)
    N = 1000
    x_train = gmm.rand_gmm(N, alpha_true, mu_true, Sigma_true)
    
    print("True parameters:")
    print("Alpha:", alpha_true)
    print("Means:\n", mu_true)
    
    # Fit the model
    gmm.fit(x_train)
    
    print("\nEstimated parameters:")
    print("Alpha:", gmm.alpha)
    print("Means:\n", gmm.mu)
    
    # Generate validation set and compute likelihood
    x_val = gmm.rand_gmm(N, alpha_true, mu_true, Sigma_true)
    train_ll = gmm.score(x_train)
    val_ll = gmm.score(x_val)
    
    print(f"\nLog-likelihood - Train: {train_ll:.2f}, Validation: {val_ll:.2f}")
    
    # Show final plot
    plt.show()

if __name__ == "__main__":
    test()