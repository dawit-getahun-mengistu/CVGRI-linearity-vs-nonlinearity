from PIL import Image
import io
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from nn import AlexNet, DeepLinearConvNet
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import argparse
from pathlib import Path
import os
import base64

from cfg import device, dataset_path
from load import CUBDataset


def get_last_conv_layer(model):
    """Helper function to get the last convolutional layer"""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def get_last_linear_layer(model):
    """Helper function to get the last linear layer"""
    last_linear = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    return last_linear


def get_feature_layer(model):
    """Helper function to get the appropriate feature layer based on model architecture"""
    if isinstance(model, AlexNet):
        # Get features from the last convolutional layer (not fc1)
        return model.layer5
    elif isinstance(model, DeepLinearConvNet):
        # Get features from the last conv layer (conv3)
        return model.conv3
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def extract_features(model, dataloader, device, use_features=True, feature_layer=None):
    """
    Extract features from a specific layer of the model for PCA or t-SNE.
    """
    features_list = []
    labels_list = []
    images_list = []
    features_tensor = None

    # Helper to set up a hook
    def hook_fn(module, input, output):
        nonlocal features_tensor
        if len(output.shape) == 4:  # Conv layer output
            # Flatten spatial dimensions
            features_tensor = output.view(output.size(0), -1)
        else:  # Linear layer output
            features_tensor = output

    # Determine which layer to hook if not specified
    if feature_layer is None:
        feature_layer = get_feature_layer(model)

    print('feature_layer: ', feature_layer)

    if feature_layer is None:
        raise ValueError(
            "Feature layer not specified or could not be determined.")

    # Register the forward hook
    hook_handle = feature_layer.register_forward_hook(hook_fn)

    model.eval()  # Set model to evaluation mode
    try:
        with torch.no_grad():
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(device)

                # Forward pass through the model
                model(batch_images)
                if features_tensor is None:
                    raise ValueError(
                        "Feature extraction failed; hook did not capture features.")

                # Convert extracted features to numpy arrays
                features_list.append(features_tensor.cpu().numpy())
                labels_list.extend(batch_labels.cpu().numpy())
                images_list.extend(batch_images.cpu().numpy())

    finally:
        # Remove the hook to avoid interference with future operations
        hook_handle.remove()

    # Combine all features, labels, and images into numpy arrays
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    images = np.array(images_list)

    print(f"Extracted features shape: {features.shape}")
    return features, labels, images
