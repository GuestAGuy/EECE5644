import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class VehicleLocalizationMAP:
    """
    Vehicle Localization using MAP estimation with range measurements
    
    Mathematical Formulation:
    MAP objective: J(x,y) = Σ[(r_i - d_i(x,y))²/(2σ_i²)] + [x²/(2σ_x²) + y²/(2σ_y²)]
    where d_i(x,y) = √((x-x_i)² + (y-y_i)²) is the distance to landmark i
    """
    
    def __init__(self, sigma_x=0.25, sigma_y=0.25, measurement_noise_std=0.3):
        """
        Initialize the localization system
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.measurement_noise_std = measurement_noise_std
    
    def generate_true_position(self):
        """Generate true vehicle position inside unit circle centered at origin"""
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 1)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y])
    
    def generate_landmarks(self, K):
        """
        Place K landmarks evenly spaced on unit circle centered at origin
        """
        angles = np.linspace(0, 2*np.pi, K, endpoint=False)
        landmarks = np.column_stack([np.cos(angles), np.sin(angles)])
        return landmarks
    
    def generate_measurements(self, true_position, landmarks):
        """
        Generate range measurements with Gaussian noise
        """
        measurements = []
        for landmark in landmarks:
            true_distance = np.linalg.norm(true_position - landmark)
            # Ensure non-negative measurements (resample if negative)
            measurement = -1
            while measurement < 0:
                noise = np.random.normal(0, self.measurement_noise_std)
                measurement = true_distance + noise
            measurements.append(measurement)
        return np.array(measurements)
    
    def map_objective(self, position, landmarks, measurements):
        """
        Compute MAP objective function value at given position
        """
        x, y = position
        
        # Prior term: -log P(x,y) ∝ x²/(2σ_x²) + y²/(2σ_y²)
        prior_term = (x**2 / (2 * self.sigma_x**2)) + (y**2 / (2 * self.sigma_y**2))
        
        # Measurement likelihood terms: -log P(r|x,y) ∝ Σ[(r_i - d_i)²/(2σ_i²)]
        measurement_terms = 0
        for i, landmark in enumerate(landmarks):
            predicted_distance = np.linalg.norm(position - landmark)
            residual = measurements[i] - predicted_distance
            measurement_terms += (residual**2) / (2 * self.measurement_noise_std**2)
        
        return prior_term + measurement_terms
    
    def find_map_estimate(self, landmarks, measurements):
        """
        Find MAP estimate by minimizing the objective function
        """
        result = minimize(self.map_objective, [0, 0], 
                         args=(landmarks, measurements), 
                         method='BFGS')
        
        return result.x if result.success else None
    
    def plot_contours(self, landmarks, true_position, measurements, K, ax):
        """
        Plot MAP objective function contours
        """
        
        # Create evaluation grid
        x_grid = np.linspace(-2, 2, 100)
        y_grid = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Evaluate objective function on grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self.map_objective([X[i,j], Y[i,j]], landmarks, measurements)
        
        # Use consistent contour levels for comparison across plots
        min_val = np.min(Z)
        max_val = np.max(Z)
        levels = np.linspace(min_val, min_val + (max_val-min_val)*0.8, 15)
        
        # Plot contours
        contour = ax.contour(X, Y, Z, levels=levels)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Mark true position and landmarks
        ax.plot(true_position[0], true_position[1], 'r+', markersize=15, 
                markeredgewidth=3, label='True Position')
        ax.plot(landmarks[:,0], landmarks[:,1], 'go', markersize=10, 
                label='Landmarks', alpha=0.7)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title(f'K = {K} Landmarks')
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return Z