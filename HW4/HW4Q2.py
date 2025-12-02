# **EECE5644 Assignment 4 - Problem 2: IMPROVED GMM Image Segmentation**
# With feature weighting and better initialization for higher accuracy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import cv2
import os

def load_and_preprocess_image(image_path, max_pixels=50000):
    """Load image with optional downsampling"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Auto-downsample
    h, w = image.shape[:2]
    if h * w > max_pixels:
        factor = np.sqrt((h * w) / max_pixels)
        new_h, new_w = int(h / factor), int(w / factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Downsampled from {h}x{w} to {new_h}x{new_w}")
    
    return image

def extract_and_weight_features(image, spatial_weight=0.3):
    """
    Extract 5D features with weighted spatial components
    Higher spatial_weight = more emphasis on spatial proximity
    Lower spatial_weight = more emphasis on color similarity
    """
    h, w = image.shape[:2]
    
    # Create coordinate grids
    rows, cols = np.mgrid[0:h, 0:w]
    
    # Extract features
    spatial_features = np.column_stack([rows.ravel(), cols.ravel()])
    color_features = image.reshape(-1, 3)
    
    # Normalize spatial features to [0,1]
    spatial_norm = spatial_features.astype(np.float32)
    spatial_norm[:, 0] = (spatial_norm[:, 0] - spatial_norm[:, 0].min()) / \
                        (spatial_norm[:, 0].max() - spatial_norm[:, 0].min() + 1e-10)
    spatial_norm[:, 1] = (spatial_norm[:, 1] - spatial_norm[:, 1].min()) / \
                        (spatial_norm[:, 1].max() - spatial_norm[:, 1].min() + 1e-10)
    
    # Normalize color features to [0,1]
    color_norm = color_features.astype(np.float32) / 255.0
    
    # Apply spatial weighting
    spatial_norm = spatial_norm * spatial_weight
    color_norm = color_norm * (1.0 - spatial_weight)
    
    # Combine features into 5D vector
    features = np.column_stack([spatial_norm, color_norm])
    
    return features, (h, w)

def improved_gmm_cross_validation(features, max_components=8, n_folds=3, random_state=42):
    """
    Improved CV with better initialization and feature weighting
    """
    print(f"Testing 2 to {max_components} components...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = []
    
    for n_components in range(2, max_components + 1):
        fold_scores = []
        
        for train_idx, val_idx in kf.split(features):
            X_train, X_val = features[train_idx], features[val_idx]
            
            # Use k-means initialization for better starting points
            if len(X_train) > 0:
                n_unique = min(n_components, len(np.unique(X_train, axis=0)))
                kmeans = KMeans(n_clusters=n_unique, random_state=random_state, n_init=3)
                initial_means = kmeans.fit(X_train).cluster_centers_
            else:
                initial_means = 'k-means++'
            
            # Fit GMM with better settings
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=random_state,
                max_iter=200,
                n_init=3,
                tol=1e-4,
                reg_covar=1e-6,
                init_params='kmeans',
                means_init=initial_means if n_components > 2 else None
            )
            
            gmm.fit(X_train)
            fold_scores.append(gmm.score(X_val))
        
        avg_score = np.mean(fold_scores)
        scores.append(avg_score)
        print(f"  Components: {n_components}, Avg LL: {avg_score:.4f}")
    
    best_n = np.argmax(scores) + 2
    best_score = scores[best_n - 2]
    
    print(f"\n Best: {best_n} components (score: {best_score:.4f})")
    return best_n, list(range(2, max_components + 1)), scores

def fit_improved_gmm(features, n_components):
    """Fit GMM with multiple strategies for better accuracy"""
    print(f"Fitting GMM with {n_components} components...")
    
    # Strategy 1: Try with full covariance first
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=300,
        n_init=5,
        tol=1e-4,
        reg_covar=1e-6,
        init_params='kmeans'
    )
    gmm.fit(features)
    print(f"  Used full covariance (log-likelihood: {gmm.score(features):.4f})")
    
    return gmm

def create_improved_segmentation(gmm, features, image_shape, use_confidence=False):
    """Create segmentation with optional confidence-based refinement"""
    print("Creating segmentation...")
    
    # Get both labels and probabilities
    labels = gmm.predict(features)
    probabilities = gmm.predict_proba(features)
    
    # Reshape to image
    segmented = labels.reshape(image_shape[0], image_shape[1])
    
    if use_confidence:
        # Get confidence scores
        confidence = np.max(probabilities, axis=1).reshape(image_shape[0], image_shape[1])
        
        # Create confidence-based segmentation
        confidence_threshold = 0.7
        low_confidence_mask = confidence < confidence_threshold
        
        # Apply median filter to smooth low-confidence areas
        if np.any(low_confidence_mask):
            from scipy.ndimage import median_filter
            smoothed = median_filter(segmented, size=3)
            segmented[low_confidence_mask] = smoothed[low_confidence_mask]
    
    # Create grayscale version for contrast
    unique_labels = np.unique(segmented)
    n_labels = len(unique_labels)
    
    if n_labels > 1:
        # Uniformly distribute grayscale values for good contrast
        grayscale_seg = (segmented - segmented.min()) / (segmented.max() - segmented.min() + 1e-10)
        grayscale_seg = (grayscale_seg * 255).astype(np.uint8)
    else:
        grayscale_seg = np.full((image_shape[0], image_shape[1]), 128, dtype=np.uint8)
    
    return grayscale_seg, labels

def plot_improved_results(original, grayscale_seg, 
                         component_range, scores, best_n,
                         spatial_weight):
    """Enhanced visualization - removed colored segmentation display"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Grayscale segmentation (now occupies both top positions for better visibility)
    axes[0, 1].imshow(grayscale_seg, cmap='gray')
    axes[0, 1].set_title('Grayscale Segmentation (GMM-based)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # CV results
    axes[1, 0].plot(component_range, scores, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=best_n, color='red', linestyle='--', linewidth=2)
    axes[1, 0].plot(best_n, scores[best_n-2], 'ro', markersize=10, 
                   label=f'Best: {best_n}')
    axes[1, 0].set_xlabel('Number of Components', fontsize=11)
    axes[1, 0].set_ylabel('Avg Validation Log-Likelihood', fontsize=11)
    axes[1, 0].set_title('Cross-Validation Results', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Information panel
    info_text = f"""
    PARAMETERS
    ----------
    Spatial weight: {spatial_weight}
    Components: {best_n}
    CV folds: 3
    
    MODEL INFO
    ----------
    Features: 5D [row, col, R, G, B]
    Normalization: [0,1] per dimension
    Spatial scaling: {spatial_weight}
    Color scaling: {1-spatial_weight:.1f}
    
    RESULTS
    -------
    Best CV score: {scores[best_n-2]:.4f}
    Image size: {original.shape[1]}x{original.shape[0]}
    """
    
    axes[1, 1].text(0.1, 0.95, info_text, transform=axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gmm_improved_segmentation.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("="*70)
    print("EECE5644 Assignment 4 - Problem 2: GMM Segmentation")
    print("="*70)
    
    # Use your image
    image_file = "42044.jpg"
    
    print(f"Loading {image_file}...")
    original = load_and_preprocess_image(image_file, max_pixels=50000)
    print(f" Loaded: {original.shape[1]}x{original.shape[0]}")
    
    # Extract features with spatial weighting
    print("\nExtracting features with adaptive weighting...")
    # Try different spatial weights to see what works best for your image
    # spatial_weight = 0.01  # Emphasizes color   (bad)
    # spatial_weight = 0.1 
    # spatial_weight = 0.4 
    # spatial_weight = 0.5 
    # spatial_weight = 0.8
    spatial_weight = 0.9   # Best
    # spatial_weight = 0.99    # emphasis spatial  (not good)
    
    features, img_shape = extract_and_weight_features(original, spatial_weight=spatial_weight)
    print(f" Created {len(features):,} feature vectors")
    print(f" Spatial weight: {spatial_weight}, Color weight: {1-spatial_weight}")
    
    # Improved cross-validation
    print("\nPerforming improved cross-validation...")
    best_n, comp_range, scores = improved_gmm_cross_validation(
        features, 
        max_components=6,
        n_folds=5,
        random_state=42
    )
    
    # Fit improved model
    print("\nFitting improved GMM...")
    final_gmm = fit_improved_gmm(features, best_n)
    
    # Create improved segmentation
    print("\nCreating improved segmentation...")
    grayscale_seg, labels = create_improved_segmentation(
        final_gmm, features, img_shape, use_confidence=True
    )
    
    # Plot results
    print("\nGenerating visualization...")
    plot_improved_results(original, grayscale_seg, 
                         comp_range, scores, best_n, spatial_weight)

if __name__ == "__main__":
    main()