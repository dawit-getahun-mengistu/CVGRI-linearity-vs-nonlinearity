import torch
import torch.nn as nn
import torchattacks
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cfg import device, dataset_path
from load import CUBDataset

from nn import AlexNet
from viz import get_model


from tqdm import tqdm


# Define mean and std used in normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


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

class_names = dataset.classes

# Reuse your existing setup code
model = get_model("alexnet2", num_classes).to(device)
# eps = 0.001
eps = 0.005
# eps = 0.002
attack = torchattacks.FGSM(model, eps=eps)

# Lists to store misclassified examples
all_orig_images = []
all_adv_images = []
all_perturbations = []
all_orig_preds = []
all_adv_preds = []

model.eval()
for images, labels in tqdm(dataloader):
    images, labels = images.to(device), labels.to(device)

    # Generate adversarial images
    adv_images = attack(images, labels)

    # Get predictions
    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    # Find misclassified examples
    misclass_idx = torch.where(orig_preds != adv_preds)[0]

    if len(misclass_idx) > 0:
        all_orig_images.append(images[misclass_idx])
        all_adv_images.append(adv_images[misclass_idx])
        all_perturbations.append(
            adv_images[misclass_idx] - images[misclass_idx])
        all_orig_preds.append(orig_preds[misclass_idx])
        all_adv_preds.append(adv_preds[misclass_idx])

if len(all_orig_images) > 0:
    # Concatenate all misclassified examples
    all_orig_images = torch.cat(all_orig_images)
    all_adv_images = torch.cat(all_adv_images)
    all_perturbations = torch.cat(all_perturbations)
    all_orig_preds = torch.cat(all_orig_preds)
    all_adv_preds = torch.cat(all_adv_preds)

    # Calculate attack success rate
    total_images = len(dataloader.dataset)
    misclass_count = len(all_orig_images)
    success_rate = (misclass_count / total_images) * 100

    print(f"Attack Success Rate: {success_rate:.2f}%")
    print(f"Total Misclassified: {misclass_count} out of {total_images}")

    # Visualization code remains the same as before, but using the concatenated tensors
    # Display first 5 misclassified examples
    num_display = min(6, len(all_orig_images))
    # num_display = len(all_orig_images)

    if num_display > 0:
        # Prepare images for visualization
        misclassified_original_images = denormalize(
            all_orig_images[:num_display], mean, std
        ).cpu().permute(0, 2, 3, 1).numpy()
        misclassified_adv_images_np = denormalize(
            all_adv_images[:num_display], mean, std
        ).cpu().permute(0, 2, 3, 1).numpy()
        misclassified_perturbations_np = (
            all_perturbations[:num_display]
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
        )

        # Normalize perturbations for visualization
        misclassified_perturbations_np = (
            (misclassified_perturbations_np - misclassified_perturbations_np.min())
            / (misclassified_perturbations_np.max() - misclassified_perturbations_np.min())
        )

        # Plot the misclassified images
        fig, axes = plt.subplots(num_display, 3, figsize=(12, num_display * 4))
        fig.suptitle(
            f"Misclassified Examples (Success Rate: {success_rate:.2f}%)", fontsize=16
        )

        for i in range(num_display):
            # Original image
            axes[i, 0].imshow(np.clip(misclassified_original_images[i], 0, 1))
            orig_class = class_names[all_orig_preds[i].item()]
            axes[i, 0].set_title(f"Original: {orig_class}")
            axes[i, 0].axis("off")

            # Perturbation
            axes[i, 1].imshow(misclassified_perturbations_np[i])
            axes[i, 1].set_title("Perturbation")
            axes[i, 1].axis("off")

            # Adversarial image
            axes[i, 2].imshow(np.clip(misclassified_adv_images_np[i], 0, 1))
            adv_class = class_names[all_adv_preds[i].item()]
            axes[i, 2].set_title(f"Adversarial: {adv_class}")
            axes[i, 2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
