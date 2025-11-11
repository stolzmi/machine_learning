"""
Demonstration script for Layer-wise Relevance Propagation and Activation Maximization

This script shows how to use LRP and Activation Maximization to analyze
the MNIST CNN model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf

from mnist_cnn_model import MNISTCNN, create_model
from mnist_lrp_activation_max import (
    LayerRelevancePropagation,
    ActivationMaximization,
    plot_lrp_visualization,
    plot_activation_maximization_grid,
    plot_optimization_progress
)


def load_model(model_path='mnist_model.pkl'):
    """Load trained MNIST model"""
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    params = checkpoint['params']
    batch_stats = checkpoint['batch_stats']
    model = create_model(num_classes=10)

    print(f"Loaded model from {model_path}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

    return model, params, batch_stats


def load_sample_images(n_samples=5):
    """Load sample MNIST images for analysis"""
    def normalize_img(data):
        image = tf.cast(data['image'], tf.float32) / 255.0
        label = data['label']
        return {'image': image, 'label': label}

    # Load test dataset
    test_ds = tfds.load('mnist', split='test', as_supervised=False)
    test_ds = test_ds.map(normalize_img)
    test_ds = test_ds.take(n_samples)

    images = []
    labels = []

    for batch in test_ds.as_numpy_iterator():
        images.append(batch['image'])
        labels.append(batch['label'])

    return np.array(images), np.array(labels)


def demo_layer_relevance_propagation(model, params, batch_stats, images, labels):
    """
    Demonstrate Layer-wise Relevance Propagation

    Args:
        model: MNIST model
        params: Model parameters
        batch_stats: Batch statistics
        images: Sample images
        labels: True labels
    """
    print("\n" + "="*70)
    print("LAYER-WISE RELEVANCE PROPAGATION (LRP) DEMONSTRATION")
    print("="*70)

    lrp = LayerRelevancePropagation(model, epsilon=1e-10)
    digit_names = [str(i) for i in range(10)]

    # Analyze each sample image
    for idx, (image, true_label) in enumerate(zip(images[:3], labels[:3])):
        print(f"\nAnalyzing image {idx + 1} (True label: {true_label})...")

        # Compute LRP
        relevance_map, predicted_class, class_score = lrp.compute_lrp(
            params, batch_stats, image
        )

        print(f"  Predicted class: {predicted_class} (score: {class_score:.3f})")
        print(f"  Relevance map shape: {relevance_map.shape}")
        print(f"  Relevance range: [{relevance_map.min():.3f}, {relevance_map.max():.3f}]")

        # Visualize
        fig = plot_lrp_visualization(
            image.squeeze(),
            relevance_map,
            predicted_class=predicted_class,
            class_score=class_score,
            digit_names=digit_names,
            save_path=f'lrp_demo_{idx}.png'
        )
        plt.close(fig)

        print(f"  Visualization saved to 'lrp_demo_{idx}.png'")

    # Compare standard LRP vs epsilon rule
    print("\nComparing LRP variants...")
    image = images[0]

    relevance_standard, pred1, score1 = lrp.compute_lrp(params, batch_stats, image)
    relevance_epsilon, pred2, score2 = lrp.compute_lrp_epsilon_rule(
        params, batch_stats, image
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    im1 = axes[1].imshow(relevance_standard, cmap='seismic')
    axes[1].set_title('LRP (Gradient-based)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(relevance_epsilon, cmap='seismic')
    axes[2].set_title('LRP (Epsilon Rule)', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('lrp_comparison.png', dpi=150)
    plt.close()

    print("  Comparison saved to 'lrp_comparison.png'")


def demo_activation_maximization_classes(model, params, batch_stats):
    """
    Demonstrate Activation Maximization for all digit classes

    Args:
        model: MNIST model
        params: Model parameters
        batch_stats: Batch statistics
    """
    print("\n" + "="*70)
    print("ACTIVATION MAXIMIZATION - CLASS VISUALIZATION")
    print("="*70)

    actmax = ActivationMaximization(model)
    digit_names = [str(i) for i in range(10)]

    # Generate images for all 10 digits
    generated_images = []
    final_scores = []
    score_histories = []

    print("\nGenerating images that maximize each digit class...")

    for class_idx in range(10):
        print(f"  Generating image for digit {class_idx}...", end=' ')

        # Generate image
        image, scores = actmax.maximize_class(
            params, batch_stats, class_idx,
            n_iterations=300,
            learning_rate=1.0,
            l2_reg=0.005,
            blur_every=10,
            blur_sigma=0.5,
            seed=42 + class_idx
        )

        generated_images.append(image)
        final_scores.append(scores[-1])
        score_histories.append(scores)

        print(f"Final score: {scores[-1]:.3f}")

    # Visualize all generated images
    fig = plot_activation_maximization_grid(
        generated_images,
        final_scores,
        class_names=digit_names,
        title="Activation Maximization: What Each Digit Class Looks Like",
        save_path='actmax_all_digits.png'
    )
    plt.close(fig)

    print("\n  Grid visualization saved to 'actmax_all_digits.png'")

    # Plot optimization progress
    fig = plot_optimization_progress(
        score_histories,
        labels=digit_names,
        save_path='actmax_progress.png'
    )
    plt.close(fig)

    print("  Optimization progress saved to 'actmax_progress.png'")


def demo_activation_maximization_neurons(model, params, batch_stats):
    """
    Demonstrate Activation Maximization for individual neurons

    Args:
        model: MNIST model
        params: Model parameters
        batch_stats: Batch statistics
    """
    print("\n" + "="*70)
    print("ACTIVATION MAXIMIZATION - NEURON VISUALIZATION")
    print("="*70)

    actmax = ActivationMaximization(model)

    # Visualize what different conv filters are looking for
    layers_to_visualize = [
        ('conv1', 8, "Conv Layer 1 (Low-level features)"),
        ('conv2', 8, "Conv Layer 2 (Mid-level features)"),
        ('conv3', 8, "Conv Layer 3 (High-level features)")
    ]

    for layer_name, n_neurons, layer_title in layers_to_visualize:
        print(f"\nVisualizing {n_neurons} neurons from {layer_name}...")

        generated_images = []
        final_scores = []
        neuron_labels = []

        for neuron_idx in range(n_neurons):
            print(f"  Neuron {neuron_idx}...", end=' ')

            image, scores = actmax.maximize_neuron(
                params, batch_stats, layer_name, neuron_idx,
                n_iterations=200,
                learning_rate=1.0,
                l2_reg=0.01,
                blur_every=5,
                blur_sigma=0.3,
                seed=42 + neuron_idx
            )

            generated_images.append(image)
            final_scores.append(scores[-1])
            neuron_labels.append(f"Filter {neuron_idx}")

            print(f"Score: {scores[-1]:.2f}")

        # Visualize
        fig = plot_activation_maximization_grid(
            generated_images,
            final_scores,
            class_names=neuron_labels,
            title=layer_title,
            save_path=f'actmax_{layer_name}_neurons.png'
        )
        plt.close(fig)

        print(f"  Saved to 'actmax_{layer_name}_neurons.png'")


def demo_combined_analysis(model, params, batch_stats, images, labels):
    """
    Demonstrate combined LRP and Activation Maximization analysis

    Args:
        model: MNIST model
        params: Model parameters
        batch_stats: Batch statistics
        images: Sample images
        labels: True labels
    """
    print("\n" + "="*70)
    print("COMBINED ANALYSIS: LRP + ACTIVATION MAXIMIZATION")
    print("="*70)

    lrp = LayerRelevancePropagation(model)
    actmax = ActivationMaximization(model)

    # Pick one example
    image = images[0]
    true_label = labels[0]

    print(f"\nAnalyzing image with true label: {true_label}")

    # Get prediction and LRP
    relevance_map, predicted_class, class_score = lrp.compute_lrp(
        params, batch_stats, image
    )

    print(f"  Predicted class: {predicted_class} (score: {class_score:.3f})")

    # Generate what the model thinks this digit should look like
    print(f"  Generating ideal '{predicted_class}' according to the model...")
    ideal_image, scores = actmax.maximize_class(
        params, batch_stats, predicted_class,
        n_iterations=300,
        learning_rate=1.0,
        l2_reg=0.005
    )

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image.squeeze(), cmap='gray')
    axes[0, 0].set_title(f'Input Image\n(True: {true_label}, Pred: {predicted_class})',
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # LRP relevance
    im = axes[0, 1].imshow(relevance_map, cmap='seismic')
    axes[0, 1].set_title('LRP: What mattered for prediction?',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Ideal generated image
    axes[1, 0].imshow(ideal_image, cmap='gray')
    axes[1, 0].set_title(f"ActMax: Model's ideal '{predicted_class}'",
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay LRP on original
    overlaid = lrp.visualize(image.squeeze(), relevance_map)
    axes[1, 1].imshow(overlaid)
    axes[1, 1].set_title('LRP Overlay on Input',
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    fig.suptitle('Combined XAI Analysis: Understanding Model Decisions',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('combined_xai_analysis.png', dpi=150)
    plt.close()

    print("  Combined analysis saved to 'combined_xai_analysis.png'")


def main():
    """Main demonstration function"""
    print("="*70)
    print("MNIST XAI DEMONSTRATION: LRP + ACTIVATION MAXIMIZATION")
    print("="*70)

    # Load model
    print("\nLoading trained MNIST model...")
    model, params, batch_stats = load_model('mnist_model.pkl')

    # Load sample images
    print("\nLoading sample images...")
    images, labels = load_sample_images(n_samples=5)
    print(f"Loaded {len(images)} sample images")

    # Demo 1: Layer-wise Relevance Propagation
    demo_layer_relevance_propagation(model, params, batch_stats, images, labels)

    # Demo 2: Activation Maximization for classes
    demo_activation_maximization_classes(model, params, batch_stats)

    # Demo 3: Activation Maximization for neurons
    demo_activation_maximization_neurons(model, params, batch_stats)

    # Demo 4: Combined analysis
    demo_combined_analysis(model, params, batch_stats, images, labels)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - lrp_demo_*.png: Individual LRP visualizations")
    print("  - lrp_comparison.png: Comparison of LRP methods")
    print("  - actmax_all_digits.png: Generated ideal digits (0-9)")
    print("  - actmax_progress.png: Optimization progress curves")
    print("  - actmax_*_neurons.png: Neuron-level visualizations")
    print("  - combined_xai_analysis.png: Combined LRP + ActMax analysis")


if __name__ == '__main__':
    main()
