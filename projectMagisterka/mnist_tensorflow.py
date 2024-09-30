import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples  # This requires `emnist` package

# Install the emnist package via pip if you haven't already:
# pip install emnist

# Step 1: Load the EMNIST dataset (Balanced version)
# It contains both digits and uppercase/lowercase letters
train_images, train_labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')

# Check dataset shape
print(f"Train Images Shape: {train_images.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}, Test Labels Shape: {test_labels.shape}")
