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


model = get_model("alexnet", num_classes).to(device)
model.eval()


def get_low_prob_indices(model, dataloader):
    model.eval()
    low_prob_inputs = []
    low_prob_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            low_prob_mask = probs.max(dim=1).values < 0.6
            low_prob_inputs.append(inputs[low_prob_mask])
            low_prob_labels.append(labels[low_prob_mask])

    low_prob_inputs = torch.cat(low_prob_inputs)
    low_prob_labels = torch.cat(low_prob_labels)
    return low_prob_inputs, low_prob_labels


def fgsm_attack(model, inputs, labels, epsilon):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_inputs = inputs + epsilon * inputs.grad.sign()
    adv_inputs = torch.clamp(adv_inputs, 0, 1)
    return adv_inputs


def fgsm_attack_with_noise(model, inputs, labels, epsilon, noise_scale=0.05):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Compute the FGSM perturbation
    grad_sign = inputs.grad.sign()

    # Add random noise to the perturbation
    noise = torch.randn_like(inputs) * noise_scale
    adv_inputs = inputs + epsilon * (grad_sign + noise)

    # Ensure the adversarial inputs are clipped within valid range
    adv_inputs = torch.clamp(adv_inputs, 0, 1)
    return adv_inputs


def compute_perturbation_magnitude(original, adversarial):
    perturbation = (adversarial - original).detach().cpu().numpy()
    return np.linalg.norm(perturbation)


# def visualize_adversarial_example(original, adversarial, index):
#     original = original.detach().cpu().numpy()
#     adversarial = adversarial.detach().cpu().numpy()
#     preturbation = adversarial - original

#     original_img = original.transpose(1, 2, 0)
#     adversarial_img = adversarial.transpose(1, 2, 0)
#     perturbation_img = preturbation.transpose(1, 2, 0)

#     plt.figure(figsize=(12, 4))

#     # Original Image
#     plt.subplot(1, 3, 1)
#     plt.imshow(
#         original_img, cmap="gray" if original_img.shape[-1] == 1 else None)
#     plt.title("Original Input")
#     plt.axis("off")

#     # Perturbation
#     plt.subplot(1, 3, 2)
#     plt.imshow(perturbation_img,
#                cmap="gray" if perturbation_img.shape[-1] == 1 else None)
#     plt.title("Perturbation")
#     plt.axis("off")

#     # Adversarial Image
#     plt.subplot(1, 3, 3)
#     plt.imshow(adversarial_img,
#                cmap="gray" if adversarial_img.shape[-1] == 1 else None)
#     plt.title("Adversarial Input")
#     plt.axis("off")

#     plt.suptitle(f"Adversarial Example {index + 1}")
#     plt.show()

# def visualize_grid(sorted_examples, original_labels, adversarial_labels, num_rows=2, num_cols=3):

#     plt.figure(figsize=(15, 5 * num_rows))

#     for i in range(num_rows * num_cols):
#         if i >= len(sorted_examples):
#             break
#         original, adversarial = sorted_examples[i]
#         original_label = original_labels[i]
#         adversarial_label = adversarial_labels[i]

#         perturbation = adversarial - original

#         # Compute perturbation for visualization
#         mean = np.array([0.485, 0.456, 0.406]).reshape(
#             1, 1, 3)  # Shape (1, 1, 3)
#         std = np.array([0.229, 0.224, 0.225]).reshape(
#             1, 1, 3)  # Shape (1, 1, 3)

#         perturbation = (adversarial - original).cpu().detach().numpy()
#         perturbation = np.transpose(perturbation, (1, 2, 0))  # (H, W, C)
#         perturbation = (perturbation * std) + mean  # Denormalize
#         perturbation_img = np.clip(perturbation, 0, 1)

#         original_img = original.transpose(1, 2, 0)
#         adversarial_img = adversarial.transpose(1, 2, 0)
#         # perturbation_img = denormalized.transpose(1, 2, 0)

#         # Plot each example
#         plt.subplot(num_rows, num_cols * 3, i * 3 + 1)
#         plt.imshow(
#             original_img, cmap="gray" if original_img.shape[-1] == 1 else None)
#         plt.title(f"Original\nLabel: {original_label}")
#         plt.axis("off")

#         plt.subplot(num_rows, num_cols * 3, i * 3 + 2)
#         plt.imshow(perturbation_img,
#                    cmap="gray" if perturbation_img.shape[-1] == 1 else None)
#         plt.title("Perturbation")
#         plt.axis("off")

#         plt.subplot(num_rows, num_cols * 3, i * 3 + 3)
#         plt.imshow(adversarial_img,
#                    cmap="gray" if adversarial_img.shape[-1] == 1 else None)
#         plt.title(f"Adversarial\nMisclassified: {adversarial_label}")
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()


def visualize_grid(sorted_examples, original_labels, adversarial_labels, num_rows=2, num_cols=3):
    plt.figure(figsize=(15, 5 * num_rows))

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for i in range(num_rows * num_cols):
        if i >= len(sorted_examples):
            break

        original, adversarial = sorted_examples[i]
        original_label = original_labels[i]
        adversarial_label = adversarial_labels[i]

        if original.ndim == 4:
            original, adversarial = original.squeeze(0), adversarial.squeeze(0)

        # Compute perturbation
        perturbation = (adversarial - original).cpu().detach().numpy()
        print("shape:", perturbation.shape)

        perturbation = np.transpose(perturbation, (1, 2, 0))  # (H, W, C)
        # perturbation = (perturbation * std) + mean  # Denormalize
        # perturbation = np.clip(perturbation, 0, 1)  # Clamp to [0, 1]

        # Prepare images
        original_img = original.cpu().detach().numpy().transpose(1, 2, 0)  # (H, W, C)
        adversarial_img = adversarial.cpu().detach().numpy().transpose(1, 2, 0)  # (H, W, C)

        # Plot original image
        plt.subplot(num_rows, num_cols * 3, i * 3 + 1)
        plt.imshow(
            original_img, cmap="gray" if original_img.shape[-1] == 1 else None)
        plt.title(f"Original\nLabel: {original_label}")
        plt.axis("off")

        # Plot perturbation
        plt.subplot(num_rows, num_cols * 3, i * 3 + 2)
        plt.imshow(
            perturbation, cmap="gray" if perturbation.shape[-1] == 1 else None)
        plt.title("Perturbation")
        plt.axis("off")

        # Plot adversarial image
        plt.subplot(num_rows, num_cols * 3, i * 3 + 3)
        plt.imshow(adversarial_img,
                   cmap="gray" if adversarial_img.shape[-1] == 1 else None)
        plt.title(f"Adversarial\nMisclassified: {adversarial_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


epsilon = 0.1
adv_examples = []

low_prob_inputs, low_prob_labels = get_low_prob_indices(model, dataloader)
print(f"Number of low probability examples: {len(low_prob_inputs)}")

for inputs, labels in zip(low_prob_inputs, low_prob_labels):
    example = fgsm_attack(
        model, inputs.unsqueeze(0), labels.unsqueeze(0), epsilon)
    # example = fgsm_attack(
    #     model, inputs.unsqueeze(0), labels.unsqueeze(0), epsilon)

    if example is not None:
        adv_examples.append(example)

print(f"Number of adversarial examples: {len(adv_examples)}")

if len(adv_examples) > 0:
    print("Adversarial Examples:", adv_examples[0], "\n")
    adv_examples = torch.cat(adv_examples)

    misclassified_examples = []
    misclassified_original = []
    misclassified_labels = []
    misclassified_preds = []

    model.eval()
    with torch.no_grad():
        outputs = model(adv_examples)
        adv_preds = outputs.argmax(dim=1)
        # print("Adversarial Predictions:", adv_preds)

        for i, pred in enumerate(adv_preds):
            if pred != low_prob_labels[i]:
                misclassified_examples.append(adv_examples[i])
                # misclassified_examples.append(
                #     denormalize(adv_examples[i], mean, std))
                misclassified_original.append(low_prob_inputs[i])
                # misclassified_original.append(
                #     denormalize(low_prob_inputs[i], mean, std))
                misclassified_labels.append(low_prob_labels[i])
                misclassified_preds.append(pred)

    print(f"Number of misclassified examples: {len(misclassified_examples)}")

    # if len(misclassified_examples) > 0:
    #     for i in range(min(5, len(misclassified_examples))):
    #         visualize_adversarial_example(
    #             low_prob_inputs[i], adv_examples[i], i)

    perturbation_magnitudes = [
        compute_perturbation_magnitude(
            misclassified_original[i], misclassified_examples[i])
        for i in range(len(misclassified_examples))
    ]
    sorted_indices = np.argsort(perturbation_magnitudes)
    sorted_misclassified_examples = [
        (misclassified_original[i], misclassified_examples[i]) for i in sorted_indices
    ]
    sorted_original_labels = [misclassified_labels[i].item()
                              for i in sorted_indices]
    sorted_adversarial_labels = [
        misclassified_preds[i].item() for i in sorted_indices]

    visualize_grid(sorted_misclassified_examples,
                   sorted_original_labels, sorted_adversarial_labels)
