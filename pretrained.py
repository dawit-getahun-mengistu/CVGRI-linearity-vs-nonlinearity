import torch
import torchvision.models as models
import os


save_dir = "./pretrained/alexnet"
os.makedirs(save_dir, exist_ok=True)

# Load AlexNet pre-trained on ImageNet
alexnet = models.alexnet(pretrained=True)

# Save the state_dict to a file
weights_path = os.path.join(save_dir, "alexnet_imagenet.pth")
torch.save(alexnet.state_dict(), weights_path)

print(f"AlexNet weights saved at: {weights_path}")
