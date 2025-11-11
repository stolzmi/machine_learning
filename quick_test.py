"""
Quick test to verify the model works correctly
Trains for just 2 epochs on a small subset of data
"""

import jax
from jax import random
import numpy as np
from train_cifar10 import create_train_state, load_cifar10_data, train_epoch, evaluate_model

print("="*60)
print("Quick Test - CIFAR-10 CNN with XAI")
print("="*60)

# Initialize
print("\n1. Initializing model...")
rng = random.PRNGKey(42)
state, model = create_train_state(rng, learning_rate=0.001)
print("   Model initialized successfully!")

# Load small dataset
print("\n2. Loading CIFAR-10 dataset...")
import tensorflow as tf
train_ds, val_ds, test_ds, info = load_cifar10_data(batch_size=64)
print(f"   Training samples: {info.splits['train'].num_examples}")
print(f"   Test samples: {info.splits['test'].num_examples}")

# Train for 2 epochs
print("\n3. Training for 2 epochs (this is just a test)...")
for epoch in range(1, 3):
    state, train_metrics = train_epoch(state, train_ds, epoch)
    val_metrics = evaluate_model(state, val_ds)

    print(f"\n   Epoch {epoch}/2")
    print(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
    print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Train full model: python train_cifar10.py")
print("2. Run XAI demo: python demo_xai.py")
