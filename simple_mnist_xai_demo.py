"""
Simple XAI demonstration with MNIST using random weights
Shows how the XAI methods work without needing a trained model
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from pathlib import Path

from mnist_cnn_model import MNISTCNN, initialize_model
from mnist_xai_visualizations import GradCAM, SaliencyMap

# MNIST digit names
MNIST_DIGITS = [str(i) for i in range(10)]


def load_sample_image():
    """Load a single MNIST image"""
    ds = tfds.load('mnist', split='test', as_supervised=False)

    for example in ds.take(1):
        image = np.array(example['image'], dtype=np.float32) / 255.0
        label = int(example['label'])
        return image, label


def main():
    print("="*60)
    print("Simple XAI Demo - MNIST")
    print("Using randomly initialized model (not trained)")
    print("="*60)

    # Initialize model with random weights
    print("\nInitializing model with random weights...")
    rng = random.PRNGKey(0)
    params, batch_stats, model = initialize_model(rng)
    print("Model initialized!")

    # Load a sample image
    print("\nLoading sample image...")
    image, true_label = load_sample_image()
    image_2d = image.squeeze(-1)  # For display
    print(f"True label: {MNIST_DIGITS[true_label]}")

    # Make prediction (will be random since model is not trained)
    print("\nMaking prediction...")
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, jnp.expand_dims(image, 0), training=False)
    pred_class = int(jnp.argmax(logits[0]))
    probs = jax.nn.softmax(logits[0])

    print(f"Predicted: {MNIST_DIGITS[pred_class]} ({float(probs[pred_class]):.2%})")
    print("\nNote: Prediction is random because model is not trained!")

    # Initialize XAI methods
    print("\n" + "="*60)
    print("Computing XAI Visualizations")
    print("="*60)

    gradcam = GradCAM(model, target_layer='conv3')
    saliency = SaliencyMap(model)

    # Compute GradCAM
    print("\n1. Computing GradCAM...")
    gradcam_heatmap, _, _ = gradcam.compute_gradcam(params, batch_stats, image)
    print("   Done!")

    # Compute Saliency Map
    print("\n2. Computing Saliency Map...")
    saliency_map, _, _ = saliency.compute_saliency(params, batch_stats, image)
    print("   Done!")

    # Create visualization
    print("\n3. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: Original and overlays
    axes[0, 0].imshow(image_2d, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gradcam.visualize(image_2d, gradcam_heatmap))
    axes[0, 1].set_title('GradCAM Overlay', fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(saliency.visualize(image_2d, saliency_map))
    axes[0, 2].set_title('Saliency Overlay', fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Heatmaps
    axes[1, 0].imshow(image_2d, cmap='gray')
    axes[1, 0].set_title(f'True: {MNIST_DIGITS[true_label]}', fontweight='bold')
    axes[1, 0].axis('off')

    im1 = axes[1, 1].imshow(gradcam_heatmap, cmap='jet')
    axes[1, 1].set_title('GradCAM Heatmap', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    im2 = axes[1, 2].imshow(saliency_map, cmap='hot')
    axes[1, 2].set_title('Saliency Heatmap', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(f'XAI Demo - Predicted: {MNIST_DIGITS[pred_class]} | True: {MNIST_DIGITS[true_label]}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir = Path('mnist_xai_results')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'simple_mnist_xai_demo.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    print(f"\n   Saved to: {save_path}")

    # Show
    plt.show()

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nWhat you see:")
    print("- GradCAM: Shows which spatial regions the model focuses on")
    print("- Saliency: Shows which pixels are most important")
    print("\nNote: Since the model is not trained, the visualizations")
    print("      show random patterns. Train the model first for")
    print("      meaningful explanations!")
    print("\nTo train the model, run: python train_mnist.py")


if __name__ == '__main__':
    main()
