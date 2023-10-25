from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import config

class DawnDataset:
    def __init__(self, root_dir, batch_size=32, num_workers=4, mode="train", transform=None) -> None:
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.domain_map = {
            'Fog':  [1, 0, 0, 0],
            'Rain': [0, 1, 0, 0],
            'Sand': [0, 0, 1, 0],
            'Snow': [0, 0, 0, 1]
        }

        self.dataset = ImageFolder(root=root_dir, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    def get_dataloader(self):
        return self.data_loader
    
    def get_domain_labels(self, folder_labels):
        return torch.tensor([self.domain_map[self.dataset.classes[label]] for label in folder_labels])



if __name__ == "__main__":

    transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = DawnDataset(root_dir=config.DAWN_DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, transform=transform)

    data_loader = dataset.get_dataloader()

    for images, labels in data_loader:
        print(images.shape)  # Should be [batch_size, 3, 128, 128]
        domain_labels = dataset.get_domain_labels(labels)
        print(domain_labels)
        break

