from data_loader import ACDCDataset
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import config
from train import train_fn
import os
from utils import save_checkpoint, save_some_examples

from torchvision import transforms
import torch
import torch.optim as optim


def main():
    # Create Datalaoder
    transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(config.IMAGE_SIZE), # is this doing anything ?
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


    val_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


    # Create StarGAN
    disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS)
    disc = disc.to(config.DEVICE)

    gen = Generator(in_channels=config.CHANNEL_IMG, feautues=64, c_dim=config.NUM_DOMAINS)
    gen = gen.to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    #BCE = nn.BCEWithLogitsLoss() # standard GAN loss


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc=disc, gen=gen, loader=train_loader, g_opt=opt_gen, d_opt=opt_disc) # could pass BCE, GradientScaler

        if config.SAVE_MODEL and epoch%5==0:
            pass
            #save_checkpoint(gen, opt_gen, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_GEN))
            #save_checkpoint(disc, opt_disc, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_DISC))

        #save_some_examples(gen, val_loader, epoch, folder=config.EVAL_DIR)


if __name__ == "__main__":
    main()
    




    