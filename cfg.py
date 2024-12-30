import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = './dataset/CUB_200_2011'


if __name__ == '__main__':
    print(device)
    print(dataset_path)
    print(torch.cuda.is_available())
