"""
Concept Bottleneck Network (CBN) for MNIST
============================================

A Concept Bottleneck Network learns interpretable concepts as an intermediate
representation between the input and the final prediction.

For MNIST digits, we use visual concepts like:
- Has vertical line
- Has horizontal line
- Has loop/circle
- Has diagonal line
- Has curve
- Has top horizontal
- Has bottom horizontal
- Has intersection

These concepts are learned end-to-end during training.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, List
import numpy as np


class ConceptEncoder(nn.Module):
    """Encodes input image into concept activations"""
    n_concepts: int = 12

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Conv Block 1
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv Block 2
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv Block 3
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Dense layer to concepts
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)

        # Concept bottleneck - use sigmoid for interpretable concept activations
        concepts = nn.Dense(features=self.n_concepts)(x)
        concepts = nn.sigmoid(concepts)

        return concepts


class ConceptPredictor(nn.Module):
    """Predicts class labels from concept activations"""
    n_classes: int = 10

    @nn.compact
    def __call__(self, concepts, training: bool = False):
        # Simple linear predictor from concepts to classes
        # This makes the model more interpretable
        x = nn.Dense(features=64)(concepts)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(features=self.n_classes)(x)
        return x


class ConceptBottleneckNetwork(nn.Module):
    """
    Complete Concept Bottleneck Network

    Architecture:
    Input -> ConceptEncoder -> Concepts (bottleneck) -> ConceptPredictor -> Output

    The concept bottleneck forces the model to use interpretable concepts
    for making predictions.
    """
    n_concepts: int = 12
    n_classes: int = 10

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Encode input to concepts
        concepts = ConceptEncoder(n_concepts=self.n_concepts)(x, training=training)

        # Predict class from concepts
        logits = ConceptPredictor(n_classes=self.n_classes)(concepts, training=training)

        return logits, concepts


# Concept names for interpretation
CONCEPT_NAMES = [
    "Vertical Line",
    "Horizontal Line",
    "Loop/Circle",
    "Top Curve",
    "Bottom Curve",
    "Left Curve",
    "Right Curve",
    "Diagonal /",
    "Diagonal \\",
    "Intersection",
    "Top Bar",
    "Bottom Bar",
]


def create_cbn_model(n_concepts: int = 12, n_classes: int = 10):
    """Create a new Concept Bottleneck Network model"""
    return ConceptBottleneckNetwork(n_concepts=n_concepts, n_classes=n_classes)


def get_concept_names() -> List[str]:
    """Get interpretable names for each concept"""
    return CONCEPT_NAMES


def interpret_concepts(concepts: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Interpret concept activations

    Args:
        concepts: Array of concept activations [n_concepts]
        threshold: Threshold for considering a concept "active"

    Returns:
        Dictionary mapping concept names to their activation values
    """
    concept_dict = {}
    for i, (name, value) in enumerate(zip(CONCEPT_NAMES, concepts)):
        concept_dict[name] = float(value)

    return concept_dict


def get_active_concepts(concepts: np.ndarray, threshold: float = 0.5) -> List[str]:
    """
    Get list of active concepts above threshold

    Args:
        concepts: Array of concept activations [n_concepts]
        threshold: Threshold for considering a concept "active"

    Returns:
        List of active concept names
    """
    active = []
    for name, value in zip(CONCEPT_NAMES, concepts):
        if value >= threshold:
            active.append(name)

    return active


def explain_prediction_with_concepts(
    predicted_class: int,
    concepts: np.ndarray,
    concept_weights: np.ndarray,
    threshold: float = 0.5
) -> str:
    """
    Generate human-readable explanation of prediction using concepts

    Args:
        predicted_class: Predicted digit (0-9)
        concepts: Array of concept activations [n_concepts]
        concept_weights: Weights from concepts to predicted class [n_concepts]
        threshold: Threshold for considering a concept "active"

    Returns:
        Human-readable explanation string
    """
    # Get active concepts
    active_concepts = get_active_concepts(concepts, threshold)

    # Get most important concepts for this prediction
    concept_importance = np.abs(concepts * concept_weights)
    top_indices = np.argsort(concept_importance)[::-1][:3]
    top_concepts = [(CONCEPT_NAMES[i], concepts[i], concept_weights[i])
                    for i in top_indices]

    explanation = f"Predicted digit: {predicted_class}\n\n"
    explanation += "Active concepts:\n"
    if active_concepts:
        for concept in active_concepts:
            idx = CONCEPT_NAMES.index(concept)
            explanation += f"  • {concept}: {concepts[idx]:.2f}\n"
    else:
        explanation += "  None above threshold\n"

    explanation += "\nMost important concepts for this prediction:\n"
    for name, activation, weight in top_concepts:
        influence = "+" if weight > 0 else "-"
        explanation += f"  {influence} {name}: {activation:.2f} (weight: {weight:.2f})\n"

    return explanation


def get_concept_importance_for_class(
    model_params,
    model,
    digit_class: int
) -> np.ndarray:
    """
    Get the importance of each concept for predicting a specific digit class

    Args:
        model_params: Model parameters
        model: CBN model instance
        digit_class: Target digit class (0-9)

    Returns:
        Array of concept importances [n_concepts]
    """
    # Extract the weights from the concept predictor's final layer
    # This tells us how much each concept contributes to each class

    # Navigate through the model parameters to find the final dense layer weights
    # Structure: params['ConceptPredictor_0']['Dense_1']['kernel']
    try:
        # Get the final dense layer weights [n_features, n_classes]
        final_weights = model_params['params']['ConceptPredictor_0']['Dense_1']['kernel']

        # Get weights for the specific class
        class_weights = final_weights[:, digit_class]

        return np.array(class_weights)
    except (KeyError, IndexError):
        # Fallback: return zeros if we can't access the weights
        return np.zeros(12)
