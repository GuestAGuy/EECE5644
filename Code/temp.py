import numpy as np
import matplotlib.pyplot as plt
from Vehicle_Localization_Map import VehicleLocalizationMAP

def main():
    np.random.seed(42)
    localizer = VehicleLocalizationMAP()
    true_position = localizer.generate_true_position()
    
    K_values = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print(f"True vehicle position: ({true_position[0]:.3f}, {true_position[1]:.3f})")
    
    for idx, K in enumerate(K_values):
        landmarks = localizer.generate_landmarks(K)
        measurements = localizer.generate_measurements(true_position, landmarks)
        map_estimate = localizer.find_map_estimate(landmarks, measurements)
        
        localizer.plot_contours(landmarks, true_position, measurements, K, axes[idx])
        
        if map_estimate is not None:
            error = np.linalg.norm(map_estimate - true_position)
            print(f"K={K}: MAP estimate error = {error:.4f}")
    
    plt.tight_layout()
    plt.savefig('map_localization_contours.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()