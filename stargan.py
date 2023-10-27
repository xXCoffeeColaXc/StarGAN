import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model import Discriminator, Generator, ConvBlock
import config
import time
from train import train_fn
from utils import save_checkpoint, save_some_examples, load_checkpoint
import os


class StarGAN():
    def __init__(self, train_loader, val_loader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.build_model()


    def train(self):
        if config.LOAD_MODEL:
            self.restore_model()
        else:
            self.gen.apply(self.weights_init_normal)
            self.disc.apply(self.weights_init_normal)

        # Start training.
        if config.LOG:
            print('[START TRAINING]')
        start_time = time.time()

        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc=self.disc, gen=self.gen, loader=self.train_loader, g_opt=self.opt_gen, d_opt=self.opt_disc, start_time=start_time) # could pass GradientScaler

            if config.SAVE_MODEL and epoch%5==0:
                self.save_model()
                
            # Save images for debugging
            save_some_examples(self.gen, self.val_loader, epoch, folder=config.EVAL_DIR)

    def test(self):
        pass

    def evaluate(self):
        pass

    def run_inference(self):
        pass

    def build_model(self):
        # Initialize generator and discriminator
        self.disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS)
        self.disc = self.disc.to(config.DEVICE)

        self.gen = Generator(in_channels=config.CHANNEL_IMG, feautues=64, c_dim=config.NUM_DOMAINS)
        self.gen = self.gen.to(config.DEVICE)

        # Optimizers
        self.opt_disc = Adam(self.disc.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
        self.opt_gen = Adam(self.gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

        if config.LOG:
            print("[MODEL BUILT]")

    def print_model(self):
        print("--- GENERATOR ---")
        print(self.gen.parameters)

        print("--- DISCRIMINATOR ---")
        print(self.disc.parameters)

    def restore_model(self):
        load_checkpoint(os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_GEN), self.gen, self.opt_gen, config.LEARNING_RATE)
        load_checkpoint(os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_DISC), self.disc, self.opt_disc, config.LEARNING_RATE)

    def save_model(self):
        save_checkpoint(self.gen, self.opt_gen, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_GEN))
        save_checkpoint(self.disc, self.opt_disc, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_DISC))

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("ConvBlock") != -1 or classname.find("ResidualBlock") != -1:
            for submodule in m.children():
                if isinstance(submodule, nn.Conv2d):
                    torch.nn.init.normal_(submodule.weight.data, 0.0, 0.02)


