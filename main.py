from torchvision import transforms
from data_loader import ACDCDataset
from torch.utils.data import DataLoader
import config
from stargan import StarGAN

def main():
    # Create Datalaoders
    transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(config.IMAGE_SIZE), # is this doing anything ?
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    train_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


    # Initialize StarGAN
    model = StarGAN(train_loader=train_loader, val_loader=val_loader)

    model.print_model()

    model.train()

if __name__ == "__main__":
    main()
    




    