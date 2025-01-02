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

from feature_extraction import extract_features


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


def visualize_embeddings(features, labels, images, model_metadata, method='pca', perplexity=30, n_components=2, save_dir=None, model_name=None):

    if features.shape[1] > 50:
        print(
            f"Applying initial PCA to reduce dimensions from {features.shape[1]} to 50")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    print(f"Applying {method.upper()} for final visualization")
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:  # t-SNE
        reducer = TSNE(n_components=n_components, perplexity=perplexity)

    embeddings = reducer.fit_transform(features)

    hover_data = {
        'Class': labels.astype(str),
        'Image': [f"<img src='data:image/png;base64,{base64.b64encode(transforms.ToPILImage()(image).tobytes()).decode()}'/>" for image in images]
    }

    if n_components == 2:
        fig = px.scatter(
            embeddings, x=0, y=1,
            color=labels.astype(str),
            labels={'color': 'Class'},
            hover_data=hover_data,
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
            hover_data=hover_data,
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

    if save_dir:
        metadata_str = "\n".join(
            [f"{key}: {value}" for key, value in model_metadata.items()])
        fig.update_layout(
            title_text=f"{method.upper()} Visualization\n{metadata_str}")

        filename = f"{model_name if model_name else ''}_{method}_{n_components}d.html"
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Saved plot to {save_path}")

    fig.show()


def create_image_hover(img_tensor):
    """Convert tensor image to base64 string for hover display"""
    # Denormalize the image if it was normalized
    img = img_tensor.cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    # Convert to PIL Image
    img_pil = transforms.ToPILImage()(img)

    # Resize for hover display (adjust size as needed)
    img_pil = img_pil.resize((100, 100), Image.Resampling.LANCZOS)

    # Convert to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Create data URI with image data
    hover_image = f'data:image/png;base64,{img_str}'

    return hover_image


def create_combined_visualization(features, labels, images, model_metadata, perplexity=30, save_dir=None, model_name=None):
    """Create a single HTML file with all visualizations and image hovers"""

    # Initial dimension reduction if needed
    if features.shape[1] > 50:
        print(
            f"Applying initial PCA to reduce dimensions from {features.shape[1]} to 50")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    # Create embeddings for both PCA and t-SNE in 2D and 3D
    print("Computing embeddings...")
    pca_2d = PCA(n_components=2).fit_transform(features)
    pca_3d = PCA(n_components=3).fit_transform(features)
    tsne_2d = TSNE(
        n_components=2, perplexity=perplexity).fit_transform(features)
    try:
        tsne_3d = TSNE(
            n_components=3, perplexity=perplexity).fit_transform(features)
    except Exception as e:
        print(f"Error during t-SNE 3D computation: {e}")

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('PCA 2D', 't-SNE 2D', 'PCA 3D', 't-SNE 3D')
    )

    # Create hover template with images
    # hovertemplate = """
    # <img src='%{customdata[0]}' width=100 height=100><br>
    # Class: %{customdata[1]}<br>
    # <extra></extra>
    # """

    # Add traces for each plot
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]

    # Process all images for hover display
    print("Processing images for hover display...")
    hover_images = [create_image_hover(img) for img in images]

    # 2D PCA
    for i, label in enumerate(unique_labels):
        mask = labels == label
        customdata = np.column_stack([
            np.array(hover_images)[mask],
            np.full(np.sum(mask), label)
        ])
        fig.add_trace(
            go.Scatter(
                x=pca_2d[mask, 0],
                y=pca_2d[mask, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(color=colors[i]),
                customdata=customdata,
                # hovertemplate=hovertemplate,
                showlegend=True
            ),
            row=1, col=1
        )

    # 2D t-SNE
    for i, label in enumerate(unique_labels):
        mask = labels == label
        # customdata = list(
        #     zip(np.array(hover_images)[mask], np.full(np.sum(mask), label)))
        # mask = labels == label
        # customdata = np.column_stack([
        #     np.array(hover_images)[mask],
        #     np.full(np.sum(mask), label)
        # ])
        fig.add_trace(
            go.Scatter(
                x=tsne_2d[mask, 0],
                y=tsne_2d[mask, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(color=colors[i]),
                customdata=customdata,
                # hovertemplate=hovertemplate,
                showlegend=False
            ),
            row=1, col=2
        )

    # 3D PCA
    for i, label in enumerate(unique_labels):
        mask = labels == label
        # customdata = np.column_stack([
        #     np.array(hover_images)[mask],
        #     np.full(np.sum(mask), label)
        # ])
        fig.add_trace(
            go.Scatter3d(
                x=pca_3d[mask, 0],
                y=pca_3d[mask, 1],
                z=pca_3d[mask, 2],
                mode='markers',
                name=f'Class {label}',
                marker=dict(color=colors[i]),
                customdata=customdata,
                # hovertemplate=hovertemplate,
                showlegend=False
            ),
            row=2, col=1
        )

    # 3D t-SNE
    for i, label in enumerate(unique_labels):
        mask = labels == label
        # customdata = np.column_stack([
        #     np.array(hover_images)[mask],
        #     np.full(np.sum(mask), label)
        # ])
        fig.add_trace(
            go.Scatter3d(
                x=tsne_3d[mask, 0],
                y=tsne_3d[mask, 1],
                z=tsne_3d[mask, 2],
                mode='markers',
                name=f'Class {label}',
                marker=dict(color=colors[i]),
                customdata=customdata,
                # hovertemplate=hovertemplate,
                showlegend=False
            ),
            row=2, col=2
        )

    # Update layout
    metadata_str = "<br>".join(
        [f"{key}: {value}" for key, value in model_metadata.items()])
    fig.update_layout(
        title_text=f"Model Embeddings Visualization<br><sub>{metadata_str}</sub>",
        height=1200,
        width=1600,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    # Save the figure
    if save_dir:
        filename = f"{model_name if model_name else ''}_combined_visualization.html"
        save_path = os.path.join(save_dir, filename)
        fig.write_html(save_path)
        print(f"Saved combined visualization to {save_path}")

    return fig


def get_model_info(model):
    """Count convolutional layers and ReLU activations in a model"""
    num_conv_layers = 0
    num_relu = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            num_conv_layers += 1
        elif isinstance(module, torch.nn.ReLU):
            num_relu += 1

    return num_conv_layers, num_relu


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

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model = load_trained_model(args.model, args.num_classes, args.checkpoint)

    model_info = get_model_info(model)
    num_conv_layers, num_relu = model_info

    model_metadata = {
        "Model": args.model,
        # "Checkpoint": args.checkpoint,
        "Num Classes": args.num_classes,
        "Batch Size": args.batch_size,
        # "Use Features": args.use_features,
        "Conv Layers": num_conv_layers,
        "ReLU Activations": num_relu
    }

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

    print("Extracting features...")
    features, labels, images = extract_features(
        model, dataloader, device, args.use_features)

    images = torch.Tensor(images)

    # print("Creating visualizations...")
    # try:
    #     visualize_embeddings(features, labels, images, model_metadata, method='pca',
    #                          n_components=2, save_dir=args.save_dir, model_name=args.model)
    #     visualize_embeddings(features, labels, images, model_metadata, method='pca',
    #                          n_components=3, save_dir=args.save_dir, model_name=args.model)
    # except Exception as e:
    #     print(f"Error during PCA visualization: {e}")

    # try:
    #     visualize_embeddings(features, labels, images, model_metadata, method='tsne', perplexity=args.perplexity,
    #                          n_components=2, save_dir=args.save_dir, model_name=args.model)
    #     visualize_embeddings(features, labels, images, model_metadata, method='tsne', perplexity=args.perplexity,
    #                          n_components=3, save_dir=args.save_dir, model_name=args.model)

    # except Exception as e:
    #     print(f"Error during t-SNE visualization: {e}")

    print("Creating combined visualization...")
    try:
        fig = create_combined_visualization(
            features, labels, images, model_metadata,
            perplexity=args.perplexity,
            save_dir=args.save_dir,
            model_name=args.model
        )
        fig.show()
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    # main()
    try:
        main()
    except Exception as e:
        print(f"Error during visualization: {e}")
