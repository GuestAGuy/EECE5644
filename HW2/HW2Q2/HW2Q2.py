import numpy as np
import matplotlib.pyplot as plt
import sys
import os

code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Code'))
if code_path not in sys.path:
    sys.path.insert(0, code_path)

from Cubic_Polynomial_Regression import CubicPolynomialRegression, average_squared_error
from Generate_Data_From_GMM import GenerateDataFromGMM

def generateData(N):
    """Generate data using GMM parameters"""
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3]
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = GenerateDataFromGMM.generate_data_from_gmm(N, gmmParameters)
    return x

def main():
    """Cubic Polynomial Regression: ML and MAP Estimation"""
    print("Cubic Polynomial Regression: ML and MAP Estimation")
    
    # Load data
    Ntrain = 100
    data = generateData(Ntrain)
    xTrain = data[0:2,:].T
    yTrain = data[2,:]
    
    Ntrain = 1000
    data = generateData(Ntrain)
    xValidate = data[0:2,:].T
    yValidate = data[2,:]
    
    # Maximum Likelihood Estimation
    model = CubicPolynomialRegression(sigma2=1.0)
    w_ml = model.ml_estimate(xTrain, yTrain)
    y_pred_ml = model.predict(xValidate)
    ase_ml = average_squared_error(yValidate, y_pred_ml)
    
    print(f"ML Validation ASE: {ase_ml:.6f}")
    
    # Maximum A Posteriori Estimation
    gamma_values = np.logspace(-3, 3, 50)
    ase_map_values = []
    weight_norms = []
    
    for gamma in gamma_values:
        model_map = CubicPolynomialRegression(sigma2=1.0)
        w_map = model_map.map_estimate(xTrain, yTrain, gamma)
        y_pred_map = model_map.predict(xValidate)
        ase_map = average_squared_error(yValidate, y_pred_map)
        ase_map_values.append(ase_map)
        weight_norms.append(np.linalg.norm(w_map))
    
    optimal_idx = np.argmin(ase_map_values)
    optimal_gamma = gamma_values[optimal_idx]
    optimal_ase = ase_map_values[optimal_idx]
    
    print(f"Optimal MAP ASE: {optimal_ase:.6f}")
    print(f"Optimal γ: {optimal_gamma:.6f}")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: ASE vs Gamma
    axes[0, 0].semilogx(gamma_values, ase_map_values, 'b-', linewidth=2, label='MAP ASE')
    axes[0, 0].axhline(y=ase_ml, color='r', linestyle='--', linewidth=2, label='ML ASE')
    axes[0, 0].axvline(x=optimal_gamma, color='g', linestyle=':', alpha=0.7)
    axes[0, 0].set_xlabel('Regularization Parameter γ')
    axes[0, 0].set_ylabel('Average Squared Error')
    axes[0, 0].set_title('Validation ASE vs Regularization Parameter γ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Weight Norm vs Gamma
    axes[0, 1].semilogx(gamma_values, weight_norms, 'g-', linewidth=2)
    axes[0, 1].axvline(x=optimal_gamma, color='g', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Regularization Parameter γ')
    axes[0, 1].set_ylabel('L2 Norm of Weight Vector')
    axes[0, 1].set_title('Weight Vector Norm vs Regularization Parameter γ')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Extreme gamma values comparison
    gamma_extreme = [1e-6, optimal_gamma, 1e6]
    ase_extreme = []
    
    for gamma in gamma_extreme:
        model_temp = CubicPolynomialRegression(sigma2=1.0)
        w_temp = model_temp.map_estimate(xTrain, yTrain, gamma)
        y_pred_temp = model_temp.predict(xValidate)
        ase_extreme.append(average_squared_error(yValidate, y_pred_temp))
    
    labels = ['γ=1e-6', f'γ={optimal_gamma:.3f}', 'γ=1e6']
    x_pos = np.arange(len(labels))
    axes[1, 0].bar(x_pos, ase_extreme, alpha=0.7, color=['red', 'green', 'blue'])
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Average Squared Error')
    axes[1, 0].set_title('ASE Comparison: Different γ Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Predictions vs True values
    model_best = CubicPolynomialRegression(sigma2=1.0)
    w_best = model_best.map_estimate(xTrain, yTrain, optimal_gamma)
    y_pred_best = model_best.predict(xValidate)
    
    axes[1, 1].scatter(yValidate, y_pred_best, alpha=0.6, s=20)
    min_val = min(yValidate.min(), y_pred_best.min())
    max_val = max(yValidate.max(), y_pred_best.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('True y values')
    axes[1, 1].set_ylabel('Predicted y values')
    axes[1, 1].set_title(f'Best MAP Predictions (γ={optimal_gamma:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cubic_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()