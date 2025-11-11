"""
Demo script for XAI visualizations on CIFAR-10
Demonstrates GradCAM and Saliency Maps on trained model
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow_datasets as tfds
from pathlib import Path

from cifar10_cnn_model import CIFAR10CNN
from xai_visualizations import GradCAM, SaliencyMap, plot_xai_comparison


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_trained_model(model_path: str = 'cifar10_model.pkl'):
    """
    Load trained model from pickle file

    Args:
        model_path: Path to saved model

    Returns:
        params: Model parameters
        batch_stats: Batch statistics
        model: Model instance
    """
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    params = checkpoint['params']
    batch_stats = checkpoint['batch_stats']

    model = CIFAR10CNN(num_classes=10)

    print(f"Model loaded from {model_path}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")

    return params, batch_stats, model


def load_sample_images(num_samples: int = 10, split: str = 'test'):
    """
    Load sample images from CIFAR-10

    Args:
        num_samples: Number of samples to load
        split: Dataset split ('train' or 'test')

    Returns:
        images: Array of images [num_samples, 32, 32, 3]
        labels: Array of labels [num_samples]
    """
    ds = tfds.load('cifar10', split=split, as_supervised=False)

    images = []
    labels = []

    for i, example in enumerate(ds.take(num_samples)):
        image = np.array(example['image'], dtype=np.float32) / 255.0
        label = int(example['label'])
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


def visualize_single_image(params, batch_stats, model, image, true_label,
                           save_dir: str = 'xai_results'):
    """
    Create comprehensive XAI visualization for a single image

    Args:
        params: Model parameters
        batch_stats: Batch statistics
        model: Model instance
        image: Input image [32, 32, 3]
        true_label: True class label
        save_dir: Directory to save visualizations
    """
    Path(save_dir).mkdir(exist_ok=True)

    # Initialize XAI methods
    gradcam = GradCAM(model, target_layer='conv3')
    saliency = SaliencyMap(model)

    print("\nComputing visualizations...")

    # Compute GradCAM
    print("  Computing GradCAM...")
    gradcam_heatmap, pred_class, grad_score = gradcam.compute_gradcam(
        params, batch_stats, image
    )

    # Compute Saliency Map
    print("  Computing Saliency Map...")
    saliency_map, _, sal_score = saliency.compute_saliency(
        params, batch_stats, image
    )

    # Compute Smooth Saliency
    print("  Computing Smooth Saliency (this may take a moment)...")
    smooth_saliency, _, smooth_score = saliency.compute_smooth_saliency(
        params, batch_stats, image, n_samples=30, noise_level=0.15
    )

    print(f"\nPrediction: {CIFAR10_CLASSES[pred_class]} (score: {grad_score:.3f})")
    print(f"True label: {CIFAR10_CLASSES[true_label]}")
    print(f"Correct: {pred_class == true_label}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original and overlays
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    gradcam_overlay = gradcam.visualize(image, gradcam_heatmap, alpha=0.4)
    ax2.imshow(gradcam_overlay)
    ax2.set_title('GradCAM Overlay', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    saliency_overlay = saliency.visualize(image, saliency_map, alpha=0.5)
    ax3.imshow(saliency_overlay)
    ax3.set_title('Saliency Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    smooth_overlay = saliency.visualize(image, smooth_saliency, alpha=0.5)
    ax4.imshow(smooth_overlay)
    ax4.set_title('Smooth Saliency Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2: Heatmaps
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(image)
    ax5.set_title('Original', fontsize=12, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    im1 = ax6.imshow(gradcam_heatmap, cmap='jet')
    ax6.set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im1, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[1, 2])
    im2 = ax7.imshow(saliency_map, cmap='hot')
    ax7.set_title('Saliency Heatmap', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im2, ax=ax7, fraction=0.046)

    ax8 = fig.add_subplot(gs[1, 3])
    im3 = ax8.imshow(smooth_saliency, cmap='hot')
    ax8.set_title('Smooth Saliency Heatmap', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im3, ax=ax8, fraction=0.046)

    # Row 3: Different target classes for GradCAM
    ax9 = fig.add_subplot(gs[2, 0])
    # GradCAM for predicted class
    gc_pred, _, _ = gradcam.compute_gradcam(params, batch_stats, image, class_idx=pred_class)
    ax9.imshow(gradcam.visualize(image, gc_pred, alpha=0.4))
    ax9.set_title(f'GradCAM: {CIFAR10_CLASSES[pred_class]}\n(Predicted)', fontsize=10)
    ax9.axis('off')

    # GradCAM for true class (if different)
    if pred_class != true_label:
        ax10 = fig.add_subplot(gs[2, 1])
        gc_true, _, _ = gradcam.compute_gradcam(params, batch_stats, image, class_idx=true_label)
        ax10.imshow(gradcam.visualize(image, gc_true, alpha=0.4))
        ax10.set_title(f'GradCAM: {CIFAR10_CLASSES[true_label]}\n(True Label)', fontsize=10)
        ax10.axis('off')

    # Add text box with prediction info
    ax11 = fig.add_subplot(gs[2, 2:])
    ax11.axis('off')

    # Get top-3 predictions
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, jnp.expand_dims(image, 0), training=False)
    probs = jax.nn.softmax(logits[0])
    top3_idx = jnp.argsort(probs)[-3:][::-1]

    info_text = f"True Label: {CIFAR10_CLASSES[true_label]}\n\n"
    info_text += "Top 3 Predictions:\n"
    for i, idx in enumerate(top3_idx):
        info_text += f"  {i+1}. {CIFAR10_CLASSES[int(idx)]}: {float(probs[idx]):.2%}\n"

    ax11.text(0.1, 0.5, info_text, fontsize=14, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Main title
    correct_str = "✓ CORRECT" if pred_class == true_label else "✗ INCORRECT"
    fig.suptitle(f'XAI Analysis - Predicted: {CIFAR10_CLASSES[pred_class]} | True: {CIFAR10_CLASSES[true_label]} | {correct_str}',
                fontsize=16, fontweight='bold')

    # Save
    save_path = Path(save_dir) / f'xai_{CIFAR10_CLASSES[true_label]}_{pred_class == true_label}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

    plt.close()


def create_batch_visualization(params, batch_stats, model, images, labels,
                               num_display: int = 8, save_dir: str = 'xai_results'):
    """
    Create grid visualization for multiple images

    Args:
        params: Model parameters
        batch_stats: Batch statistics
        model: Model instance
        images: Array of images [N, 32, 32, 3]
        labels: Array of labels [N]
        num_display: Number of images to display
        save_dir: Directory to save visualizations
    """
    Path(save_dir).mkdir(exist_ok=True)

    gradcam = GradCAM(model, target_layer='conv3')

    num_display = min(num_display, len(images))
    fig, axes = plt.subplots(num_display, 3, figsize=(12, 4 * num_display))

    if num_display == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_display):
        image = images[i]
        true_label = labels[i]

        # Compute GradCAM
        heatmap, pred_class, score = gradcam.compute_gradcam(
            params, batch_stats, image
        )

        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original\nTrue: {CIFAR10_CLASSES[true_label]}', fontsize=10)
        axes[i, 0].axis('off')

        # GradCAM overlay
        overlay = gradcam.visualize(image, heatmap, alpha=0.4)
        axes[i, 1].imshow(overlay)
        correct = "✓" if pred_class == true_label else "✗"
        axes[i, 1].set_title(f'GradCAM {correct}\nPred: {CIFAR10_CLASSES[pred_class]}', fontsize=10)
        axes[i, 1].axis('off')

        # Heatmap
        axes[i, 2].imshow(heatmap, cmap='jet')
        axes[i, 2].set_title(f'Heatmap\nScore: {score:.3f}', fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = Path(save_dir) / 'batch_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBatch visualization saved to {save_path}")
    plt.close()


def analyze_misclassifications(params, batch_stats, model, num_samples: int = 50,
                               save_dir: str = 'xai_results'):
    """
    Find and visualize misclassified examples

    Args:
        params: Model parameters
        batch_stats: Batch statistics
        model: Model instance
        num_samples: Number of test samples to check
        save_dir: Directory to save visualizations
    """
    print("\nSearching for misclassifications...")

    images, labels = load_sample_images(num_samples=num_samples, split='test')

    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []

    variables = {'params': params, 'batch_stats': batch_stats}

    for image, label in zip(images, labels):
        logits = model.apply(variables, jnp.expand_dims(image, 0), training=False)
        pred = int(jnp.argmax(logits[0]))

        if pred != label:
            misclassified_images.append(image)
            misclassified_labels.append(label)
            predicted_labels.append(pred)

    if len(misclassified_images) == 0:
        print("No misclassifications found in the sample!")
        return

    print(f"Found {len(misclassified_images)} misclassified images")

    # Visualize first few
    num_viz = min(4, len(misclassified_images))

    for i in range(num_viz):
        print(f"\nAnalyzing misclassification {i+1}/{num_viz}...")
        visualize_single_image(
            params, batch_stats, model,
            misclassified_images[i],
            misclassified_labels[i],
            save_dir=save_dir
        )


def main():
    """Main demo function"""
    print("="*60)
    print("CIFAR-10 CNN - XAI Visualization Demo")
    print("="*60)

    # Check if model exists
    model_path = 'cifar10_model.pkl'
    if not Path(model_path).exists():
        print(f"\nError: Model file '{model_path}' not found!")
        print("Please train the model first using: python train_cifar10.py")
        return

    # Load model
    print("\nLoading model...")
    params, batch_stats, model = load_trained_model(model_path)

    # Load sample images
    print("\nLoading sample images...")
    images, labels = load_sample_images(num_samples=10, split='test')

    # Create output directory
    save_dir = 'xai_results'
    Path(save_dir).mkdir(exist_ok=True)

    # 1. Visualize individual images
    print("\n" + "="*60)
    print("1. Creating detailed XAI visualizations for individual images")
    print("="*60)

    for i in range(min(3, len(images))):
        print(f"\nProcessing image {i+1}/3...")
        visualize_single_image(params, batch_stats, model, images[i], labels[i], save_dir)

    # 2. Create batch visualization
    print("\n" + "="*60)
    print("2. Creating batch visualization")
    print("="*60)
    create_batch_visualization(params, batch_stats, model, images, labels,
                              num_display=min(6, len(images)), save_dir=save_dir)

    # 3. Analyze misclassifications
    print("\n" + "="*60)
    print("3. Analyzing misclassifications")
    print("="*60)
    analyze_misclassifications(params, batch_stats, model, num_samples=100, save_dir=save_dir)

    print("\n" + "="*60)
    print("Demo complete! Check the 'xai_results' directory for visualizations.")
    print("="*60)


if __name__ == '__main__':
    main()
