import numpy as np
import matplotlib.pyplot as plt
import sys
import os
code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Code'))
if code_path not in sys.path:
    sys.path.insert(0, code_path)
    
from Vehicle_Localization_Map import VehicleLocalizationMAP

def main():
    localizer = VehicleLocalizationMAP()
    true_position = localizer.generate_true_position()
    
    K_values = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print(f"True vehicle position: ({true_position[0]:.3f}, {true_position[1]:.3f})")
    
    # Store errors for the accuracy plot
    errors = []
    
    for idx, K in enumerate(K_values):
        landmarks = localizer.generate_landmarks(K)
        measurements = localizer.generate_measurements(true_position, landmarks)
        map_estimate = localizer.find_map_estimate(landmarks, measurements)
        
        localizer.plot_contours(landmarks, true_position, measurements, K, axes[idx])
        
        if map_estimate is not None:
            error = np.linalg.norm(map_estimate - true_position)
            errors.append(error)
            print(f"K={K}: MAP estimate error = {error:.4f}")
        else:
            errors.append(np.nan)
    
    plt.tight_layout()
    plt.savefig('map_localization_contours.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate accuracy vs number of landmarks plot
    """Plot accuracy improvement as number of landmarks increases"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(K_values, errors, 'bo-', linewidth=2, markersize=8, label='Estimation Error')
    plt.xlabel('Number of Landmarks (K)')
    plt.ylabel('Estimation Error')
    plt.title('Localization Accuracy vs Number of Landmarks')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_values)
    
    # Add some annotations
    for i, (K, error) in enumerate(zip(K_values, errors)):
        plt.annotate(f'{error:.3f}', (K, error), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_vs_landmarks.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAccuracy Analysis:")
    print(f"True position: ({true_position[0]:.3f}, {true_position[1]:.3f})")
    for K, error in zip(K_values, errors):
        print(f"K={K}: Error = {error:.4f}")

if __name__ == "__main__":
    main()