from nn import AlexNet, DeepLinearConvNet
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import argparse
from pathlib import Path
import os

from cfg import device, dataset_path
from load import CUBDataset


def get_model(model_name, num_classes):
    models = {
        'alexnet': AlexNet(num_classes),
        'lnrdeepconv': DeepLinearConvNet(3, num_classes)
    }
    return models.get(model_name.lower())


def load_trained_model(model_name, num_classes, checkpoint_path):
    model = get_model(model_name, num_classes)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


class FeatureExtractor:
    def __init__(self, model, use_features=True):
        self.model = model
        self.use_features = use_features
        self.features = None

        if self.use_features:
            def hook(module, input, output):
                # For CNN features, flatten the spatial dimensions
                if len(output.shape) == 4:  # Conv layer output
                    self.features = output.mean(
                        dim=[2, 3])  # Global average pooling
                else:  # FC layer output
                    self.features = output

            # Register hook at appropriate layer based on model type
            if isinstance(model, AlexNet):
                # Get last conv layer features for AlexNet
                for module in model.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        last_conv = module
                last_conv.register_forward_hook(hook)

            elif isinstance(model, DeepLinearConvNet):
                # For DeepLinearConvNet, get features from last layer before classification
                # Assuming the model has sequential layers - adjust based on your architecture
                found = False
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        last_linear = module
                        found = True
                if found:
                    last_linear.register_forward_hook(hook)
                print("visualization: Found: ", found)

    def extract_features(self, x):
        with torch.no_grad():
            if self.use_features:
                # Run forward pass to trigger hook
                self.model(x)
                return self.features
            else:
                # Get logits from the final layer
                return self.model(x)

# def extract_features(model, dataloader, device, use_features=True):
#     """Extract features from either the last layer or the final features layer."""
#     features = []
#     labels = []

#     with torch.no_grad():
#         for images, batch_labels in dataloader:
#             images = images.to(device)

#             if use_features:
#                 # Get features before the final classification layer
#                 if isinstance(model, AlexNet):
#                     features_batch = model.features(images)
#                     features_batch = model.avgpool(features_batch)
#                     features_batch = torch.flatten(features_batch, 1)
#                     features_batch = model.classifier[:-1](features_batch)
#                 else:  # DeepLinearConvNet
#                     features_batch = model.features(images)
#                     features_batch = features_batch.view(
#                         features_batch.size(0), -1)
#             else:
#                 # Get logits from the final layer
#                 features_batch = model(images)

#             features.append(features_batch.cpu().numpy())
#             labels.extend(batch_labels.numpy())

#     return np.vstack(features), np.array(labels)


def extract_features(model, dataloader, device, use_features=True):
    """Extract features using the feature extractor."""
    extractor = FeatureExtractor(model, use_features)
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            features_batch = extractor.extract_features(images)
            features.append(features_batch.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.vstack(features), np.array(labels)


def visualize_embeddings(features, labels, method='pca', perplexity=30, n_components=2, save_dir=None, model_name=None):
    """Reduce dimensionality and visualize the embeddings with Plotly."""
    # First apply PCA for initial dimensionality reduction if needed
    if features.shape[1] > 50:
        print(
            f"Applying initial PCA to reduce dimensions from {features.shape[1]} to 50")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    # Apply final dimensionality reduction
    print(f"Applying {method.upper()} for final visualization")
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:  # t-SNE
        reducer = TSNE(n_components=n_components, perplexity=perplexity)

    embeddings = reducer.fit_transform(features)

    if n_components == 2:
        fig = px.scatter(
            embeddings, x=0, y=1,
            color=labels.astype(str),
            labels={'color': 'Class'},
            title=f"{method.upper()} 2D Visualization",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2"
        )
    else:
        fig = px.scatter_3d(
            embeddings, x=0, y=1, z=2,
            color=labels.astype(str),
            labels={'color': 'Class'},
            title=f"{method.upper()} 3D Visualization",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method.upper()} Component 1",
                yaxis_title=f"{method.upper()} Component 2",
                zaxis_title=f"{method.upper()} Component 3",
            )
        )

    # Save the plot if save_dir is provided
    if save_dir:
        filename = f"{model_name if model_name else ''}_{method}_{n_components}d.html"
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Saved plot to {save_path}")

    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize model embeddings')
    parser.add_argument('--model', type=str, required=True, choices=['alexnet', 'lnrdeepconv'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=dataset_path,
                        help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--use_features', action='store_true',
                        help='Use features before final layer instead of logits')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='Perplexity parameter for t-SNE')

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load the model
    model = load_trained_model(args.model, args.num_classes, args.checkpoint)

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = CUBDataset(root_dir=args.data_dir,
                         transform=transform,
                         num_classes=args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Extract features
    print("Extracting features...")
    features, labels = extract_features(
        model, dataloader, device, args.use_features)

    print("Features shape: ", features.shape, "Labels shape: ", labels.shape)
    print(f"features type: {type(features)}, labels type: {type(labels)}")
    print(f"features: {features[0]}, labels: {labels[0]}")

    # Create visualizations
    print("Creating visualizations...")
    print("Creating visualizations...")
    try:
        visualize_embeddings(features, labels, method='pca',
                             n_components=2, save_dir=args.save_dir, model_name=args.model)
        visualize_embeddings(features, labels, method='pca',
                             n_components=3, save_dir=args.save_dir, model_name=args.model)
    except Exception as e:
        print(f"Error during PCA visualization: {e}")

    try:
        visualize_embeddings(features, labels, method='tsne', perplexity=args.perplexity,
                             n_components=2, save_dir=args.save_dir, model_name=args.model)
        visualize_embeddings(features, labels, method='tsne', perplexity=args.perplexity,
                             n_components=3, save_dir=args.save_dir, model_name=args.model)
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during visualization: {e}")
