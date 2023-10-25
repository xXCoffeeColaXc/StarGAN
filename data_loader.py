from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import config

# Eventually delete Dawn Dataset, merge the two into one
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

class ACDCDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        
        # Define the weather conditions and corresponding labels
        self.conditions = ['fog', 'night', 'rain', 'snow', 'daytime']
        self.condition_labels = {
            'fog':    [1, 0, 0, 0, 0], # this should be float, might fuck up indexing ? try out
            'night':  [0, 1, 0, 0, 0],
            'rain':   [0, 0, 1, 0, 0],
            'snow':   [0, 0, 0, 1, 0],
            'daytime':[0, 0, 0, 0, 1]
        }

        self.transform = transform

        self.preprocess()

    def preprocess(self):
        # Collect all the image paths and corresponding labels
        self.img_paths = []
        self.labels = []
        for condition in self.conditions:
            condition_path = os.path.join(self.root_dir, condition, self.mode)
            for folder in os.listdir(condition_path):
                if not folder.endswith('_ref'):  # Exclude the '_ref' folders
                    folder_path = os.path.join(condition_path, folder)
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):  # Assuming images are .jpg or .png
                            self.img_paths.append(os.path.join(folder_path, img_file))
                            self.labels.append(self.condition_labels[condition])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label


if __name__ == "__main__":

    transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    acdc_transform = transforms.Compose([
            transforms.Resize(128, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(128), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    '''
    # DAWN dataset
    dataset = DawnDataset(root_dir=config.DAWN_DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, transform=transform)
    data_loader = dataset.get_dataloader()

    for images, labels in data_loader:
        print(images.shape)  # Should be [batch_size, 3, 128, 128]
        domain_labels = dataset.get_domain_labels(labels)
        print(domain_labels)
        break
    '''

    acdc_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=acdc_transform, mode='train')
    acdc_dataloader = DataLoader(acdc_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

