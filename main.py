from data_loader import DawnDataset
from model import Generator, Discriminator
import config

from torchvision import transforms
import torch

if __name__ == "__main__":

    # Create Datalaoder
    transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = DawnDataset(root_dir=config.TRAIN_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, transform=transform)

    data_loader = dataset.get_dataloader()

    # Create StarGAN
    disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS)
    disc = disc.to(config.DEVICE)

    gen = Generator(in_channels=config.CHANNEL_IMG, feautues=64, c_dim=config.NUM_DOMAINS)
    gen = gen.to(config.DEVICE)

    # Preprocess input data
    x,y = next(iter(data_loader))

    label_org = dataset.get_domain_labels(y)
    rand_idx = torch.randperm(label_org.size(0)) # Generate target domain labels randomly.
    label_trg = label_org[rand_idx]

    c_org = label_org.clone()
    c_trg = label_trg.clone()

    x_real = x.to(config.DEVICE)                # Input images.
    c_org = c_org.to(config.DEVICE)             # Original domain labels.
    c_trg = c_trg.to(config.DEVICE)             # Target domain labels.
    label_org = label_org.to(config.DEVICE)     # Labels for computing classification loss.
    label_trg = label_trg.to(config.DEVICE)     # Labels for computing classification loss.


    # Predictions
    src, cls = disc(x_real)
   
    print(src[0])
    print(cls[0])

    x_fake = gen(x_real, c_trg)

    print(x_fake.shape)
