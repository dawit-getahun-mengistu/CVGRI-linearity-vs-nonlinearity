import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px


def process_raw_images(dataloader, device):
    """Process raw images into a format suitable for PCA/t-SNE."""
    features = []
    labels = []

    for images, batch_labels in dataloader:
        # Move batch to device
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        # Reshape: (batch_size, pixels)
        batch_features = images.view(images.size(0), -1)

        # Back to CPU for numpy conversion
        features.append(batch_features.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())

    return np.vstack(features), np.array(labels)


def visualize_embeddings(features, labels, method='pca', perplexity=30, n_components=2):
    """Reduce dimensionality and visualize the embeddings with Plotly."""
    # First apply PCA for initial dimensionality reduction
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
            embeddings, x=0, y=1, color=labels.astype(str),
            labels={'color': 'Class'}, title=f"{method.upper()} 2D Visualization",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            xaxis_title=f"{method.upper()} Component 1", yaxis_title=f"{method.upper()} Component 2")
    elif n_components == 3:
        fig = px.scatter_3d(
            embeddings, x=0, y=1, z=2, color=labels.astype(str),
            labels={'color': 'Class'}, title=f"{method.upper()} 3D Visualization",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method.upper()} Component 1",
                yaxis_title=f"{method.upper()} Component 2",
                zaxis_title=f"{method.upper()} Component 3",
            )
        )
    fig.show()
