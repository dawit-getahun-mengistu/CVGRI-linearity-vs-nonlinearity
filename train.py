from nn import AlexNet, DeepLinearConvNet
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.models as models
import random
import os
import argparse
from pathlib import Path
import json

from cfg import device, dataset_path
from load import CUBDataset


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_weights = model.state_dict()


def get_train_valid_test_loader(data_dir=dataset_path, num_classes=2, batch_size=32, random_seed=42):
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CUBDataset(
        root_dir=data_dir, transform=transform, num_classes=num_classes)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.Random(random_seed).shuffle(indices)

    train_split = int(train_ratio * dataset_size)
    valid_split = train_split + int(valid_ratio * dataset_size)

    train_indices = indices[:train_split]
    valid_indices = indices[train_split:valid_split]
    test_indices = indices[valid_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, valid_loader, test_loader


def get_model(model_name, num_classes, pretrained=False):
    models_dict = {
        'alexnet': AlexNet(num_classes) if not pretrained else models.alexnet(pretrained=True),
        'lnrdeepconv': DeepLinearConvNet(3, num_classes)
    }

    model = models_dict.get(model_name.lower())
    if pretrained and model_name.lower() == 'alexnet':
        # Modify the classifier for the number of classes in the dataset
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes)

    return model


def save_model(model, model_name, optimizer, train_history, save_dir='model_checkpoints'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': train_history
    }

    save_path = os.path.join(save_dir, f'{model_name}_best.pth')
    torch.save(checkpoint, save_path)

    # Save training history as JSON for easy viewing
    history_path = os.path.join(save_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=4)


def train(args):
    train_loader, valid_loader, test_loader = get_train_valid_test_loader(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        random_seed=args.seed
    )

    model = get_model(args.model, args.num_classes, args.pretrained)
    if model is None:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )

    early_stopping = EarlyStopping(patience=args.patience)
    training_history = {
        'train_loss': [],
        'val_accuracy': [],
        'best_val_accuracy': 0.0
    }

    best_model_state = None
    best_optimizer_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                print(f"Skipping step due to non-finite loss: {loss.item()}")
                continue

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            val_loss = 0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(valid_loader)
            training_history['val_accuracy'].append(val_accuracy)

            if val_accuracy > training_history['best_val_accuracy']:
                training_history['best_val_accuracy'] = val_accuracy
                best_model_state = model.state_dict().copy()
                best_optimizer_state = optimizer.state_dict().copy()

            print(f"Epoch [{epoch+1}/{args.epochs}] - "
                  f"Training Loss: {avg_train_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.2f}%")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    print("Training completed successfully.")

    # Load the best model states
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)

    # Save the best model only once at the end
    save_model(model, args.model, optimizer, training_history)

    # Test phase with best model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Train image classification models')
    parser.add_argument('--model', type=str, required=True, choices=['alexnet', 'lnrdeepconv'],
                        help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, default=dataset_path,
                        help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained AlexNet model')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
