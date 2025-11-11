"""
Simple CBN training using downloaded MNIST data (no TensorFlow required)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import pickle
from typing import Tuple, Any
from mnist_cbn_model import create_cbn_model, CONCEPT_NAMES
import time
import urllib.request
import gzip


def download_mnist():
    """Download MNIST dataset from alternative mirror"""
    # Use alternative mirror (GitHub)
    base_url = 'https://github.com/cvdfoundation/mnist/raw/main/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    print("Downloading MNIST dataset...")
    data = {}

    for filename in files:
        url = base_url + filename
        print(f"  Downloading {filename}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                compressed_data = response.read()
                data[filename] = gzip.decompress(compressed_data)
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
            # Try alternative source
            alt_url = f'https://storage.googleapis.com/cvdf-datasets/mnist/{filename}'
            print(f"  Trying alternative source...")
            with urllib.request.urlopen(alt_url, timeout=30) as response:
                compressed_data = response.read()
                data[filename] = gzip.decompress(compressed_data)

    # Parse training images
    train_images = np.frombuffer(data['train-images-idx3-ubyte.gz'], np.uint8, offset=16)
    train_images = train_images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    # Parse training labels
    train_labels = np.frombuffer(data['train-labels-idx1-ubyte.gz'], np.uint8, offset=8)

    # Parse test images
    test_images = np.frombuffer(data['t10k-images-idx3-ubyte.gz'], np.uint8, offset=16)
    test_images = test_images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    # Parse test labels
    test_labels = np.frombuffer(data['t10k-labels-idx1-ubyte.gz'], np.uint8, offset=8)

    print(f"  Train images: {train_images.shape}")
    print(f"  Train labels: {train_labels.shape}")
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")

    return (train_images, train_labels), (test_images, test_labels)


def create_batches(images, labels, batch_size, shuffle=True):
    """Create batches from dataset"""
    n = len(images)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices], labels[batch_indices]


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
def train_step(state: TrainState, images, labels, rng):
    """Single training step"""
    def loss_fn(params):
        # Forward pass with training mode
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (logits, concepts), updates = state.apply_fn(
            variables,
            images,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': rng}  # Provide RNG for dropout
        )

        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        # Compute accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)

        return loss, (accuracy, updates['batch_stats'])

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (accuracy, batch_stats)), grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads, batch_stats=batch_stats)

    return state, loss, accuracy


@jax.jit
def eval_step(state: TrainState, images, labels):
    """Single evaluation step"""
    # Forward pass in eval mode
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, concepts = state.apply_fn(variables, images, training=False)

    # Compute metrics
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    return loss, accuracy


def train_cbn(
    n_epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    save_path: str = 'mnist_cbn_model.pkl',
    seed: int = 42
):
    """Train Concept Bottleneck Network on MNIST"""
    print("=" * 70)
    print("Training Concept Bottleneck Network on MNIST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Concepts: {len(CONCEPT_NAMES)}")
    print(f"  Seed: {seed}\n")

    # Set random seed
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Download and load data
    (train_images, train_labels), (test_images, test_labels) = download_mnist()

    # Create training state
    print("\nInitializing model...")
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

        # Train for one epoch
        train_losses = []
        train_accs = []

        for batch_images, batch_labels in create_batches(train_images, train_labels, batch_size):
            rng, step_rng = jax.random.split(rng)
            batch_images = jnp.array(batch_images)
            batch_labels = jnp.array(batch_labels)

            state, loss, acc = train_step(state, batch_images, batch_labels, step_rng)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)

        # Evaluate on test set
        test_losses = []
        test_accs = []

        for batch_images, batch_labels in create_batches(test_images, test_labels, batch_size, shuffle=False):
            batch_images = jnp.array(batch_images)
            batch_labels = jnp.array(batch_labels)

            loss, acc = eval_step(state, batch_images, batch_labels)
            test_losses.append(float(loss))
            test_accs.append(float(acc))

        test_loss = np.mean(test_losses)
        test_acc = np.mean(test_accs)

        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = state

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch:2d}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")

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


if __name__ == '__main__':
    # Train the model
    state, test_acc = train_cbn(
        n_epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        save_path='mnist_cbn_model.pkl',
        seed=42
    )

    print("\nTraining complete! Model saved to 'mnist_cbn_model.pkl'")
