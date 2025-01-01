import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import random

from cfg import device, dataset_path
from utils import process_raw_images, visualize_embeddings


class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=None, random_state=42):
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

        if num_classes is not None:
            # Pick classes
            unique_classes = list(set(self.labels))

            selected_classes = random.Random(random_state).sample(
                unique_classes, num_classes)
            print(f"Selected classes: {selected_classes}")

            # Filter images and labels for the selected classes
            filtered_indices = [i for i, label in enumerate(
                self.labels) if label in selected_classes]
            self.image_paths = [self.image_paths[i] for i in filtered_indices]
            self.labels = [self.labels[i] for i in filtered_indices]

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


def main():
    print(f"Using device: {device}")

    # Define transformations
    # transform = transforms.Compose([
    #     transforms.Resize((227, 227)),
    #     transforms.ToTensor()
    # ])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    num_classes = 2
    dataset = CUBDataset(root_dir=dataset_path,
                         transform=transform, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Process raw images
    print("Processing raw images...")
    features, labels = process_raw_images(dataloader, device)

    # Visualize using both PCA and t-SNE
    print("Creating visualizations...")
    visualize_embeddings(features, labels, method='pca', n_components=2)
    visualize_embeddings(features, labels, method='pca', n_components=3)
    visualize_embeddings(features, labels, method='tsne',
                         perplexity=30, n_components=2)
    visualize_embeddings(features, labels, method='tsne',
                         perplexity=30, n_components=3)


if __name__ == "__main__":
    main()
