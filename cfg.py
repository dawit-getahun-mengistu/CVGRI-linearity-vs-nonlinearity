import torch
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = './dataset/CUB_200_2011'

# AlexNet Normalization: taken from Pytorch: https://pytorch.org/hub/pytorch_vision_alexnet/
alexnet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
),


if __name__ == '__main__':
    print(device)
    print(dataset_path)
    print(torch.cuda.is_available())
