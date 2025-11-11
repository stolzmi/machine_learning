"""
MNIST Shape and Feature Analysis Tool
Interprets which shapes, strokes, and features lead to digit classifications

This tool helps answer questions like:
- Why does the model think this is a "7"?
- What shape features distinguish "4" from "9"?
- Which strokes are most important for recognizing "8"?
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import tensorflow_datasets as tfds
from pathlib import Path

from mnist_cnn_model import MNISTCNN
from mnist_xai_visualizations import GradCAM, SaliencyMap


MNIST_DIGITS = [str(i) for i in range(10)]


def load_trained_model(model_path: str = 'mnist_model.pkl'):
    """Load trained MNIST model"""
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    params = checkpoint['params']
    batch_stats = checkpoint['batch_stats']
    model = MNISTCNN(num_classes=10)

    return params, batch_stats, model


def analyze_shape_importance(params, batch_stats, model, image, class_idx=None):
    """
    Analyze which shapes/regions are important for classification

    Returns importance scores for different regions and features
    """
    # Ensure correct shape
    if image.ndim == 2:
        image = jnp.expand_dims(image, -1)
    if image.ndim == 3:
        image = jnp.expand_dims(image, 0)

    # Get prediction
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, image, training=False)
    predicted_class = int(jnp.argmax(logits[0]))

    if class_idx is None:
        class_idx = predicted_class

    # Compute XAI
    gradcam = GradCAM(model, target_layer='conv3')
    saliency = SaliencyMap(model)

    gradcam_map, _, _ = gradcam.compute_gradcam(params, batch_stats, image.squeeze(0))
    saliency_map, _, _ = saliency.compute_saliency(params, batch_stats, image.squeeze(0))

    return {
        'gradcam': gradcam_map,
        'saliency': saliency_map,
        'predicted_class': predicted_class,
        'logits': logits[0],
        'probs': jax.nn.softmax(logits[0])
    }


def identify_key_regions(image, saliency_map, threshold=0.5):
    """
    Identify key regions (shapes/strokes) based on saliency

    Divides image into quadrants and analyzes importance
    """
    h, w = saliency_map.shape

    # Define regions
    regions = {
        'top': saliency_map[0:h//2, :],
        'bottom': saliency_map[h//2:, :],
        'left': saliency_map[:, 0:w//2],
        'right': saliency_map[:, w//2:],
        'center': saliency_map[h//4:3*h//4, w//4:3*w//4],
        'top_left': saliency_map[0:h//2, 0:w//2],
        'top_right': saliency_map[0:h//2, w//2:],
        'bottom_left': saliency_map[h//2:, 0:w//2],
        'bottom_right': saliency_map[h//2:, w//2:]
    }

    # Compute importance scores
    region_scores = {}
    for name, region in regions.items():
        # Average saliency in region
        score = float(np.mean(region))
        # Percentage of high-importance pixels
        high_importance = float(np.mean(region > threshold))
        region_scores[name] = {
            'avg_importance': score,
            'high_importance_ratio': high_importance
        }

    return region_scores


def analyze_stroke_features(image, saliency_map):
    """
    Analyze specific stroke features:
    - Horizontal strokes
    - Vertical strokes
    - Curves/loops
    - Intersections
    """
    # Ensure 2D
    if image.ndim == 3:
        image = image.squeeze(-1)

    h, w = image.shape

    # Detect horizontal strokes (look at middle rows)
    horizontal_importance = np.mean([
        np.mean(saliency_map[h//4, :]),
        np.mean(saliency_map[h//2, :]),
        np.mean(saliency_map[3*h//4, :])
    ])

    # Detect vertical strokes (look at middle columns)
    vertical_importance = np.mean([
        np.mean(saliency_map[:, w//4]),
        np.mean(saliency_map[:, w//2]),
        np.mean(saliency_map[:, 3*w//4])
    ])

    # Detect curves (corners and circular regions)
    corners = [
        saliency_map[0:h//3, 0:w//3],      # top-left
        saliency_map[0:h//3, 2*w//3:],     # top-right
        saliency_map[2*h//3:, 0:w//3],     # bottom-left
        saliency_map[2*h//3:, 2*w//3:]     # bottom-right
    ]
    curve_importance = np.mean([np.mean(corner) for corner in corners])

    # Detect intersections (center region)
    intersection_importance = np.mean(
        saliency_map[h//3:2*h//3, w//3:2*w//3]
    )

    return {
        'horizontal_strokes': float(horizontal_importance),
        'vertical_strokes': float(vertical_importance),
        'curves': float(curve_importance),
        'intersections': float(intersection_importance)
    }


def compare_against_all_classes(params, batch_stats, model, image):
    """
    Compute what features the model looks for when considering each digit

    This shows: "If I wanted to classify this as a 7, where would I look?"
    """
    if image.ndim == 2:
        image = jnp.expand_dims(image, -1)

    gradcam = GradCAM(model, target_layer='conv3')

    class_heatmaps = {}
    class_scores = {}

    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, jnp.expand_dims(image, 0), training=False)
    probs = jax.nn.softmax(logits[0])

    for digit in range(10):
        heatmap, _, score = gradcam.compute_gradcam(
            params, batch_stats, image, class_idx=digit
        )
        class_heatmaps[digit] = heatmap
        class_scores[digit] = {
            'logit': float(logits[0, digit]),
            'probability': float(probs[digit])
        }

    return class_heatmaps, class_scores


def visualize_shape_analysis(image, analysis_result, region_scores, stroke_features,
                             save_path=None):
    """
    Create comprehensive visualization of shape analysis
    """
    if image.ndim == 3:
        image_2d = image.squeeze(-1)
    else:
        image_2d = image

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    predicted = analysis_result['predicted_class']
    probs = analysis_result['probs']

    # Row 1: Original + XAI visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_2d, cmap='gray')
    ax1.set_title(f'Original Image\nPredicted: {predicted}',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im1 = ax2.imshow(analysis_result['gradcam'], cmap='jet')
    ax2.set_title('GradCAM\n(Spatial Importance)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im1, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    im2 = ax3.imshow(analysis_result['saliency'], cmap='hot')
    ax3.set_title('Saliency Map\n(Pixel Importance)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im2, ax=ax3, fraction=0.046)

    # Show image with region overlay
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(image_2d, cmap='gray')
    # Draw region boundaries
    h, w = image_2d.shape
    ax4.axhline(y=h//2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axvline(x=w//2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_title('Region Division', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Row 2: Region importance analysis
    ax5 = fig.add_subplot(gs[1, 0:2])
    regions = ['top', 'bottom', 'left', 'right', 'center']
    scores = [region_scores[r]['avg_importance'] for r in regions]
    bars = ax5.barh(regions, scores, color='steelblue')
    ax5.set_xlabel('Average Importance', fontsize=11)
    ax5.set_title('Region Importance Analysis', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 1)
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax5.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=10)

    # Stroke feature analysis
    ax6 = fig.add_subplot(gs[1, 2:])
    features = ['horizontal\nstrokes', 'vertical\nstrokes', 'curves', 'intersections']
    feature_scores = [
        stroke_features['horizontal_strokes'],
        stroke_features['vertical_strokes'],
        stroke_features['curves'],
        stroke_features['intersections']
    ]
    bars = ax6.bar(features, feature_scores, color='coral')
    ax6.set_ylabel('Importance Score', fontsize=11)
    ax6.set_title('Stroke Feature Importance', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 1)
    # Add value labels
    for bar, score in zip(bars, feature_scores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Row 3: Prediction confidence and shape interpretation
    ax7 = fig.add_subplot(gs[2, 0:2])
    top5_idx = jnp.argsort(probs)[-5:][::-1]
    top5_labels = [MNIST_DIGITS[int(i)] for i in top5_idx]
    top5_probs = [float(probs[i]) for i in top5_idx]

    bars = ax7.barh(top5_labels, top5_probs, color='lightgreen')
    bars[0].set_color('darkgreen')  # Highlight prediction
    ax7.set_xlabel('Probability', fontsize=11)
    ax7.set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
    ax7.set_xlim(0, 1)
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
        ax7.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=10)

    # Textual interpretation
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')

    # Generate interpretation text
    interpretation = generate_interpretation(
        predicted, region_scores, stroke_features, probs
    )

    ax8.text(0.05, 0.95, interpretation,
            transform=ax8.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    ax8.set_title('Shape Interpretation', fontsize=12, fontweight='bold')

    plt.suptitle(f'Shape Analysis: Digit "{predicted}"',
                fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_interpretation(predicted_digit, region_scores, stroke_features, probs):
    """
    Generate human-readable interpretation of what shapes led to classification
    """
    interpretation = f"Classification: {predicted_digit}\n"
    interpretation += f"Confidence: {float(probs[predicted_digit]):.1%}\n\n"

    interpretation += "Key Shape Features:\n"
    interpretation += "=" * 40 + "\n\n"

    # Region analysis
    top_region = max(region_scores.items(), key=lambda x: x[1]['avg_importance'])
    interpretation += f"✓ Most important region: {top_region[0]}\n"
    interpretation += f"  (importance: {top_region[1]['avg_importance']:.3f})\n\n"

    # Stroke analysis
    stroke_names = {
        'horizontal_strokes': 'Horizontal strokes',
        'vertical_strokes': 'Vertical strokes',
        'curves': 'Curves/loops',
        'intersections': 'Intersections'
    }

    top_stroke = max(stroke_features.items(), key=lambda x: x[1])
    interpretation += f"✓ Dominant feature: {stroke_names[top_stroke[0]]}\n"
    interpretation += f"  (importance: {top_stroke[1]:.3f})\n\n"

    # Digit-specific interpretation
    digit_interpretations = {
        0: "Looking for: circular loop, uniform regions",
        1: "Looking for: vertical stroke, minimal curves",
        2: "Looking for: top curve, bottom horizontal",
        3: "Looking for: stacked curves, right-side focus",
        4: "Looking for: intersection point, angled strokes",
        5: "Looking for: top horizontal, bottom curve",
        6: "Looking for: bottom loop, top curve",
        7: "Looking for: top horizontal, diagonal stroke",
        8: "Looking for: top and bottom loops, center",
        9: "Looking for: top loop, vertical stem"
    }

    interpretation += f"Expected for '{predicted_digit}':\n"
    interpretation += f"  {digit_interpretations[predicted_digit]}\n"

    return interpretation


def analyze_digit_confusion(params, batch_stats, model, image,
                            save_dir='mnist_xai_results'):
    """
    Analyze which features might cause confusion between digits
    """
    Path(save_dir).mkdir(exist_ok=True)

    if image.ndim == 2:
        image_2d = image
        image = jnp.expand_dims(image, -1)
    else:
        image_2d = image.squeeze(-1)

    # Get all class heatmaps
    class_heatmaps, class_scores = compare_against_all_classes(
        params, batch_stats, model, image
    )

    # Get predicted and top competitors
    sorted_classes = sorted(class_scores.items(),
                          key=lambda x: x[1]['probability'],
                          reverse=True)

    predicted = sorted_classes[0][0]
    top_3 = sorted_classes[:3]

    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for idx, digit in enumerate(range(10)):
        ax = axes[idx // 5, idx % 5]

        # Overlay heatmap
        ax.imshow(image_2d, cmap='gray', alpha=0.6)
        im = ax.imshow(class_heatmaps[digit], cmap='jet', alpha=0.4)

        prob = class_scores[digit]['probability']
        title = f'As "{digit}"\n{prob:.1%}'

        if digit == predicted:
            title += ' ✓'
            ax.set_title(title, fontweight='bold',
                        fontsize=11, color='green')
        else:
            ax.set_title(title, fontsize=10)

        ax.axis('off')

    plt.suptitle('What features lead to each classification?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path(save_dir) / 'digit_confusion_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion analysis to {save_path}")

    plt.close()

    return top_3, class_heatmaps


def main():
    """Main demo"""
    print("="*60)
    print("MNIST Shape Analysis Tool")
    print("Interpreting which shapes lead to classifications")
    print("="*60)

    # Load model
    model_path = 'mnist_model.pkl'
    if not Path(model_path).exists():
        print(f"\nError: Model '{model_path}' not found!")
        print("Train the model first: python train_mnist.py")
        return

    print("\nLoading model...")
    params, batch_stats, model = load_trained_model(model_path)

    # Load sample images
    print("Loading sample images...")
    ds = tfds.load('mnist', split='test', as_supervised=False)

    save_dir = Path('mnist_xai_results')
    save_dir.mkdir(exist_ok=True)

    # Analyze several digits
    for i, example in enumerate(ds.take(5)):
        image = np.array(example['image'], dtype=np.float32) / 255.0
        true_label = int(example['label'])

        print(f"\n{'='*60}")
        print(f"Analyzing digit {i+1}/5 (True label: {true_label})")
        print(f"{'='*60}")

        # Perform analysis
        analysis = analyze_shape_importance(params, batch_stats, model, image)
        region_scores = identify_key_regions(image, analysis['saliency'])
        stroke_features = analyze_stroke_features(image, analysis['saliency'])

        # Print analysis
        print(f"\nPredicted: {analysis['predicted_class']}")
        print(f"Confidence: {float(analysis['probs'][analysis['predicted_class']]):.1%}")

        print("\nTop regions:")
        sorted_regions = sorted(region_scores.items(),
                              key=lambda x: x[1]['avg_importance'],
                              reverse=True)
        for region, scores in sorted_regions[:3]:
            print(f"  {region:15s}: {scores['avg_importance']:.3f}")

        print("\nStroke features:")
        for feature, score in stroke_features.items():
            print(f"  {feature:20s}: {score:.3f}")

        # Visualize
        save_path = save_dir / f'shape_analysis_digit_{i+1}.png'
        visualize_shape_analysis(
            image, analysis, region_scores, stroke_features, save_path
        )
        print(f"\nSaved visualization to {save_path}")

        # Confusion analysis for first digit
        if i == 0:
            print("\nPerforming confusion analysis...")
            top_3, _ = analyze_digit_confusion(
                params, batch_stats, model, image, save_dir
            )

            print("\nTop 3 classifications:")
            for digit, scores in top_3:
                print(f"  {digit}: {scores['probability']:.1%}")

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Check {save_dir}/ for visualizations")
    print("="*60)


if __name__ == '__main__':
    main()
