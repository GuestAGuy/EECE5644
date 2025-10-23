import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gaussian_classifier import GaussianClassifier
import os

def load_wine_data():
    """Load and prepare the white wine quality dataset"""
    print("Loading white wine dataset...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_filenames = [
        'winequality-white.csv',
        'winequality.csv',
        'winequality_white.csv'
    ]
    
    file_path = None
    for filename in possible_filenames:
        test_path = os.path.join(script_dir, filename)
        if os.path.exists(test_path):
            file_path = test_path
            print(f"Found file: {filename}")
            break
    
    if file_path is None:
        print("File not found! Please make sure the wine quality dataset is in the same folder.")
        raise FileNotFoundError(f"Could not find wine quality dataset in: {script_dir}")
    
    wine_data = pd.read_csv(file_path, delimiter=';')
    
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol']
    
    X = wine_data[feature_columns].values
    y = wine_data['quality'].values
    feature_names = feature_columns
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    unique_qualities, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for quality, count in zip(unique_qualities, counts):
        print(f"  Quality {quality}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_names, wine_data

def compute_confusion_matrix(labels, decisions):
    """Calculate confusion matrix where entry (d,l) is P(D=d|L=l)"""
    all_classes = sorted(set(labels) | set(decisions))
    L = len(all_classes)
    
    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}
    
    confusion_matrix = np.zeros((L, L))
    class_priors = np.zeros(L)
    
    for true_label in all_classes:
        label_indices = np.where(labels == true_label)[0]
        N_l = len(label_indices)
        true_idx = class_to_index[true_label]
        
        for decision_label in all_classes:
            N_dl = np.sum(decisions[label_indices] == decision_label)
            decision_idx = class_to_index[decision_label]
            confusion_matrix[decision_idx, true_idx] = N_dl / N_l if N_l > 0 else 0
        
        class_priors[true_idx] = N_l / len(labels)
    
    return confusion_matrix, class_priors, all_classes

def evaluate_classifier(clf, X, y, dataset_name=""):
    """Test classifier and compute performance metrics"""
    y_pred = clf.predict(X)
    
    cm, class_priors, unique_labels = compute_confusion_matrix(y, y_pred)
    
    p_correct_overall = 0
    for j, true_label in enumerate(unique_labels):
        true_idx = j
        if true_label in y_pred:
            pred_idx = list(unique_labels).index(true_label)
            p_correct_overall += cm[pred_idx, true_idx] * class_priors[true_idx]
        else:
            p_correct_overall += 0 * class_priors[true_idx]
    
    p_error_overall = 1 - p_correct_overall
    
    return p_correct_overall, p_error_overall, cm, y_pred, class_priors, unique_labels

def plot_confusion_matrix(cm, unique_labels, dataset_name=""):
    """Create visual heatmap of confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, [f'L={l}' for l in unique_labels], rotation=45)
    plt.yticks(tick_marks, [f'D={l}' for l in unique_labels])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.4f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.xlabel('True Label (L)')
    plt.ylabel('Decision (D)')
    plt.title(f'Confusion Matrix - {dataset_name}\nP(D=i|L=j)')
    plt.tight_layout()
    plt.show(block=False)

def print_confusion_matrix_table(cm, unique_labels):
    """Display confusion matrix in readable table format"""
    print("\nConfusion Matrix P(D=i|L=j):")
    print("       ", end="")
    for label in unique_labels:
        print(f"L={label:2d}   ", end="")
    print("\n     " + "-" * (8 * len(unique_labels) + 1))
    
    for i, dec_label in enumerate(unique_labels):
        print(f"D={dec_label} |", end="")
        for j, true_label in enumerate(unique_labels):
            print(f" {cm[i, j]:6.4f}", end="")
        print()

def print_performance_table(cm, class_priors, unique_labels):
    """Show detailed performance breakdown by class"""
    print(f"\nPer-Class Performance:")
    print(f"Class | Correct Classification | Probability of Error")
    print(f"------|------------------------|--------------------")
    
    for j, true_label in enumerate(unique_labels):
        true_idx = j
        if true_label in unique_labels:
            pred_idx = list(unique_labels).index(true_label)
            p_correct_class = cm[pred_idx, true_idx]
        else:
            p_correct_class = 0
            
        p_error_class = 1 - p_correct_class
        
        print(f"  {true_label}   |        {p_correct_class:6.4f}          |      {p_error_class:6.4f}")

def visualize_pca(X, y, feature_names, dataset_name=""):
    """Visualize data using first 2 principal components"""
    print("\nPerforming PCA visualization...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"\nPCA Component Analysis:")
    
    pc1_weights = pca.components_[0]
    pc1_feature_importance = sorted(zip(feature_names, pc1_weights), 
                                   key=lambda x: abs(x[1]), reverse=True)
    
    print(f"PC1 ({(pca.explained_variance_ratio_[0]*100):.1f}% variance) is mainly:")
    for feature, weight in pc1_feature_importance[:3]:
        direction = "positive" if weight > 0 else "negative"
        print(f"  - {feature}: {weight:+.3f} ({direction} contribution)")
    
    pc2_weights = pca.components_[1]
    pc2_feature_importance = sorted(zip(feature_names, pc2_weights),
                                   key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nPC2 ({(pca.explained_variance_ratio_[1]*100):.1f}% variance) is mainly:")
    for feature, weight in pc2_feature_importance[:3]:
        direction = "positive" if weight > 0 else "negative"
        print(f"  - {feature}: {weight:+.3f} ({direction} contribution)")
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                         alpha=0.7, s=30, edgecolor='k', linewidth=0.3)
    plt.colorbar(scatter, label='Wine Quality')
    
    pc1_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"
    pc2_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%})"
    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)
    
    plt.title(f'PCA Visualization - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")

def main():
    """Run complete wine quality classification analysis"""
    print("WHITE WINE QUALITY ANALYSIS")
    
    X, y, feature_names, wine_data = load_wine_data()
    
    print(f"\nTraining Gaussian Classifier on {X.shape[0]} samples...")
    clf = GaussianClassifier(alpha=0.01)
    clf.fit(X, y)
    
    correct, error, cm, y_pred, priors, labels = evaluate_classifier(
        clf, X, y, "Full Dataset"
    )
    
    print(f"Full Dataset:")
    print(f"  Correct Classification: {correct:.4f}")
    print(f"  Error Probability: {error:.4f}")
    print(f"  Errors: {np.sum(y != y_pred)}/{len(y)}")
    
    print_confusion_matrix_table(cm, labels)
    print_performance_table(cm, priors, labels)
    plot_confusion_matrix(cm, labels, "White Wine Quality")
    
    visualize_pca(X, y, feature_names, "White Wine Quality")
    
    # print(f"\nMost Common Confusions:")
    # for true_label in [5, 6, 7]:
    #     pred_counts = {}
    #     for pred_label in labels:
    #         count = np.sum((y == true_label) & (y_pred == pred_label))
    #         if count > 0 and pred_label != true_label:
    #             pred_counts[pred_label] = count
    #     if pred_counts:
    #         most_common = max(pred_counts, key=pred_counts.get)
    #         print(f"  True {true_label} → Pred {most_common}: {pred_counts[most_common]} samples")

if __name__ == "__main__":
    main()