import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from cfg import device, dataset_path


class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        images_file = os.path.join(root_dir, 'images.txt')
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')

        with open(images_file, 'r') as f:
            self.image_paths = [line.strip().split()[1]
                                for line in f.readlines()]

        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip().split()[1]) -
                           1 for line in f.readlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        return image, label


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

        # back to CPU for numpy conversion
        features.append(batch_features.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())

    return np.vstack(features), np.array(labels)


def visualize_embeddings(features, labels, method='pca', perplexity=30, n_components=2):
    """Reduce dimensionality and visualize the embeddings."""

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

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                          c=labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'CUB Dataset Raw Image Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def main():
    print(f"Using device: {device}")

    # Using smaller size for raw pixel visualization
    transform = transforms.Compose([

        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = CUBDataset(root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Process raw images
    print("Processing raw images...")
    features, labels = process_raw_images(dataloader, device)

    # Visualize using both PCA and t-SNE
    print("Creating visualizations...")
    visualize_embeddings(features, labels, method='pca')
    visualize_embeddings(features, labels, method='tsne', perplexity=30)


if __name__ == "__main__":
    main()
