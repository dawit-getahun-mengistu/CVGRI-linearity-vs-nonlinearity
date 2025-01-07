import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from math import ceil

from cfg import device, dataset_path
from load import CUBDataset
from viz import get_model
from tqdm import tqdm

# Define mean and std used in normalization
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Adversarial Example Generation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., 'alexnet', 'lnrdeepconv')")
    parser.add_argument(
        "--epsilons", nargs="*", type=float, default=[0, 0.01, 0.1, 0.15],
        help="List of epsilon values for adversarial attacks"
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--output_dir", type=str, default="adversarial_examples",
                        help="Directory to save visualizations")
    return parser.parse_args()


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


def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def visualize_grid(sorted_examples, original_labels, adversarial_labels, epsilon, output_dir, model_name, num_rows=2, num_cols=3):
    num_examples = len(sorted_examples)
    num_pages = ceil(num_examples / (num_rows * num_cols))

    for page in range(num_pages):
        plt.figure(figsize=(15, 5 * num_rows))

        start_idx = page * num_rows * num_cols
        end_idx = min(start_idx + num_rows * num_cols, num_examples)

        for i, idx in enumerate(range(start_idx, end_idx)):
            original, adversarial = sorted_examples[idx]
            original_label = original_labels[idx]
            adversarial_label = adversarial_labels[idx]

            if original.ndim == 4:
                original, adversarial = original.squeeze(
                    0), adversarial.squeeze(0)

            # Compute perturbation
            perturbation = (adversarial - original).cpu().detach().numpy()
            # perturbation = np.transpose(perturbation, (1, 2, 0))
            perturbation = np.transpose(perturbation * 0.5 + 0.5, (1, 2, 0))

            # Prepare images
            original_img = original.cpu().detach().numpy().transpose(1, 2, 0)
            adversarial_img = adversarial.cpu().detach().numpy().transpose(1, 2, 0)

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
        page_path = f"{output_dir}/{model_name}/epsilon_{epsilon}/adversarial_examples_page_{page + 1}.png"
        os.makedirs(os.path.dirname(page_path), exist_ok=True)
        plt.savefig(page_path)
        plt.close()


def main():
    args = parse_arguments()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

    dataset = CUBDataset(root_dir=dataset_path,
                         transform=transform, num_classes=2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = get_model(args.model, num_classes=2).to(device)
    model.eval()

    # Get low-probability examples
    low_prob_inputs, low_prob_labels = get_low_prob_indices(model, dataloader)
    print(f"Number of low probability examples: {len(low_prob_inputs)}")

    # Process each epsilon
    accuracies = {}
    for epsilon in args.epsilons:
        print(f"Processing epsilon: {epsilon}")

        adv_examples = []
        for inputs, labels in zip(low_prob_inputs, low_prob_labels):
            example = fgsm_attack(model, inputs.unsqueeze(
                0), labels.unsqueeze(0), epsilon)
            if example is not None:
                adv_examples.append(example)

        adv_examples = torch.cat(adv_examples)
        print(f"Number of adversarial examples: {len(adv_examples)}")

        # Evaluate model on adversarial examples
        adv_dataloader = DataLoader(torch.utils.data.TensorDataset(adv_examples, low_prob_labels),
                                    batch_size=args.batch_size, shuffle=False)
        accuracy = compute_accuracy(model, adv_dataloader)
        accuracies[epsilon] = accuracy
        print(f"Accuracy for epsilon {epsilon}: {accuracy * 100:.2f}%")

        # Visualize examples
        misclassified_examples = []
        misclassified_original = []
        misclassified_labels = []
        misclassified_preds = []

        with torch.no_grad():
            outputs = model(adv_examples)
            adv_preds = outputs.argmax(dim=1)

            for i, pred in enumerate(adv_preds):
                if pred != low_prob_labels[i]:
                    misclassified_examples.append(adv_examples[i])
                    misclassified_original.append(low_prob_inputs[i])
                    misclassified_labels.append(low_prob_labels[i])
                    misclassified_preds.append(pred)

        perturbation_magnitudes = [
            (misclassified_examples[i] -
             misclassified_original[i]).norm().item()
            for i in range(len(misclassified_examples))
        ]
        sorted_indices = np.argsort(perturbation_magnitudes)
        sorted_misclassified_examples = [
            (misclassified_original[i], misclassified_examples[i]) for i in sorted_indices
        ]
        sorted_original_labels = [
            misclassified_labels[i].item() for i in sorted_indices]
        sorted_adversarial_labels = [
            misclassified_preds[i].item() for i in sorted_indices]

        visualize_grid(sorted_misclassified_examples, sorted_original_labels,
                       sorted_adversarial_labels, epsilon, args.output_dir, args.model)

    # Print final summary
    print("Accuracies for different epsilons:")
    for epsilon, accuracy in accuracies.items():
        print(f"  Epsilon {epsilon}: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
