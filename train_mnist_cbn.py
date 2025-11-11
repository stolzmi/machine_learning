"""
Train Concept Bottleneck Network on MNIST
==========================================

This script trains a Concept Bottleneck Network (CBN) on the MNIST dataset.
The CBN learns interpretable concepts as an intermediate representation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import tensorflow_datasets as tfds
import pickle
from typing import Tuple, Dict, Any
from mnist_cbn_model import create_cbn_model, CONCEPT_NAMES
import time
import tensorflow as tf 


def load_mnist_data(batch_size: int = 128):
    """Load and preprocess MNIST dataset"""
    # Load dataset
    ds_train = tfds.load('mnist', split='train', shuffle_files=True)
    ds_test = tfds.load('mnist', split='test', shuffle_files=True)

    def preprocess(sample):
        image = sample['image']
        label = sample['label']
        # Normalize to [0, 1] using TensorFlow operations
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Prepare training data
    ds_train = ds_train.map(
        lambda x: preprocess(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(10000)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Prepare test data
    ds_test = ds_test.map(
        lambda x: preprocess(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics for BatchNorm"""
    batch_stats: Any = None


def create_train_state(rng, learning_rate: float = 1e-3):
    """Create initial training state"""
    model = create_cbn_model(n_concepts=12, n_classes=10)

    # Initialize model
    dummy_input = jnp.ones((1, 28, 28, 1))
    variables = model.init(rng, dummy_input, training=False)

    # Separate parameters and batch statistics
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )


@jax.jit
def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], rng):
    """Single training step"""
    images, labels = batch

    # Split RNG for dropout
    dropout_rng, new_rng = jax.random.split(rng)

    def loss_fn(params):
        # Forward pass with training mode
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (logits, concepts), updates = state.apply_fn(
            variables,
            images,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )

        # Cross-entropy loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        # Concept sparsity regularization (L1) - encourages sparse activations
        # We want concepts to be either clearly active (>0.7) or inactive (<0.3)
        concept_sparsity = jnp.mean(jnp.minimum(concepts, 1 - concepts))

        # Concept diversity regularization - encourages different concepts for different digits
        # This prevents all concepts from being active for all inputs
        concept_diversity = -jnp.var(concepts, axis=0).mean()

        # Total loss with regularization
        lambda_sparsity = 0.01  # Sparsity weight
        lambda_diversity = 0.005  # Diversity weight

        loss = ce_loss + lambda_sparsity * concept_sparsity + lambda_diversity * concept_diversity

        # Compute accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)

        return loss, (accuracy, updates['batch_stats'])

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, batch_stats)), grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads, batch_stats=batch_stats)

    return state, {'loss': loss, 'accuracy': accuracy}


@jax.jit
def eval_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]):
    """Single evaluation step"""
    images, labels = batch

    # Forward pass in eval mode
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, concepts = state.apply_fn(variables, images, training=False)

    # Compute metrics
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    return {'loss': loss, 'accuracy': accuracy}


def train_epoch(state: TrainState, train_ds, rng):
    """Train for one epoch"""
    batch_metrics = []

    for batch in train_ds.as_numpy_iterator():
        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step(state, batch, step_rng)
        batch_metrics.append(metrics)

    # Aggregate metrics
    epoch_metrics = {
        k: np.mean([m[k] for m in batch_metrics])
        for k in batch_metrics[0].keys()
    }

    return state, epoch_metrics, rng


def evaluate(state: TrainState, test_ds):
    """Evaluate on test set"""
    batch_metrics = []

    for batch in test_ds.as_numpy_iterator():
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)

    # Aggregate metrics
    eval_metrics = {
        k: np.mean([m[k] for m in batch_metrics])
        for k in batch_metrics[0].keys()
    }

    return eval_metrics


def train_cbn(
    n_epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    save_path: str = 'mnist_cbn_model.pkl',
    seed: int = 42
):
    """
    Train Concept Bottleneck Network on MNIST

    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        save_path: Path to save trained model
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("Training Concept Bottleneck Network on MNIST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Concepts: {len(CONCEPT_NAMES)}")
    print(f"  Seed: {seed}\n")

    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Load data
    print("Loading MNIST dataset...")
    # Need to import tensorflow for the data loading
    import tensorflow as tf
    train_ds, test_ds = load_mnist_data(batch_size)
    print("Dataset loaded successfully!\n")

    # Create training state
    print("Initializing model...")
    state = create_train_state(init_rng, learning_rate)
    print("Model initialized!\n")

    # Print concept names
    print("Concept names:")
    for i, name in enumerate(CONCEPT_NAMES):
        print(f"  {i+1:2d}. {name}")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 70)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        # Train
        state, train_metrics, rng = train_epoch(state, train_ds, rng)

        # Evaluate
        test_metrics = evaluate(state, test_ds)

        # Track best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_state = state

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch:2d}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f}")

    print("-" * 70)
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")

    # Save best model
    print(f"\nSaving model to {save_path}...")
    model_data = {
        'params': best_state.params,
        'batch_stats': best_state.batch_stats,
        'test_accuracy': best_test_acc,
        'n_concepts': len(CONCEPT_NAMES),
        'concept_names': CONCEPT_NAMES,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")
    print("=" * 70)

    return best_state, best_test_acc


def load_cbn_model(path: str = 'mnist_cbn_model.pkl'):
    """Load trained CBN model"""
    with open(path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


if __name__ == '__main__':
    # Train the model with more epochs for better concept learning
    state, test_acc = train_cbn(
        n_epochs=50,  # Increased from 20 to allow concepts to stabilize
        batch_size=128,
        learning_rate=1e-3,
        save_path='mnist_cbn_model.pkl',
        seed=42
    )

    print("\nTraining complete! Model saved to 'mnist_cbn_model.pkl'")
