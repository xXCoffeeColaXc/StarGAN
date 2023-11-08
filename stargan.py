import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from modules import Discriminator, Generator, ConvBlock
import config
import time
from train import train_fn
from utils import *
import wandb

class StarGAN():
    def __init__(self, train_loader, val_loader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        if config.ENABLE_LOGGING:
            self.setup_logger()
        self.build_model()


    def train(self):
        if config.LOAD_MODEL:
            self.restore_model()
        else:
            self.gen.apply(self.weights_init_normal) # TODO try without this
            self.disc.apply(self.weights_init_normal)
        
        # Start training.
        if config.ENABLE_LOGGING:
            print('[START TRAINING]')
        start_time = time.time()

        for epoch in range(1, config.NUM_EPOCHS):
            train_fn(disc=self.disc, gen=self.gen, loader=self.train_loader, g_opt=self.opt_gen, d_opt=self.opt_disc, start_time=start_time, epoch=epoch) # could pass GradientScaler

            if config.SAVE_MODEL and epoch%5==0:
                self.save_model()
                
            # Save images for debugging
            if epoch%5==0:
                save_some_examples(self.gen, self.val_loader, epoch, folder=config.OUTPUT_IMG_DIR)


        # Finish the WandB run when you're done training
        wandb.finish()

    def test(self):
        pass

    def evaluate(self):
        pass

    def run_inference(self):
        pass

    def build_model(self):
        # Initialize generator and discriminator
        self.disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS, repeat_num=config.D_DEPTH)
        self.disc = self.disc.to(config.DEVICE)

        self.gen = Generator(in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS, repeat_num=config.G_BOTTLENECK_DEPTH)
        self.gen = self.gen.to(config.DEVICE)

        # Optimizers TODO try AdamW
        self.opt_disc = Adam(self.disc.parameters(), lr=config.D_LR, betas=(config.BETA1, config.BETA2))
        self.opt_gen = Adam(self.gen.parameters(), lr=config.G_LR, betas=(config.BETA1, config.BETA2))

        if config.ENABLE_LOGGING:
            print("[MODEL BUILT]")

    def print_model(self):
        print("--- GENERATOR ---")
        print(self.gen.parameters)

        print("--- DISCRIMINATOR ---")
        print(self.disc.parameters)

    def restore_model(self):
        load_checkpoint(os.path.join(config.OUTPUT_MODELS_DIR, config.CHECKPOINT_GEN), self.gen, self.opt_gen, config.G_LR)
        load_checkpoint(os.path.join(config.OUTPUT_MODELS_DIR, config.CHECKPOINT_DISC), self.disc, self.opt_disc, config.D_LR)

    def save_model(self):
        save_checkpoint(self.gen, self.opt_gen, filename=os.path.join(config.OUTPUT_MODELS_DIR, config.CHECKPOINT_GEN))
        save_checkpoint(self.disc,self.opt_disc,filename=os.path.join(config.OUTPUT_MODELS_DIR, config.CHECKPOINT_DISC))

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("ConvBlock") != -1 or classname.find("ResidualBlock") != -1:
            for submodule in m.children():
                if isinstance(submodule, nn.Conv2d):
                    torch.nn.init.normal_(submodule.weight.data, 0.0, 0.02)

    def setup_logger(self):
        # Initialize WandB
        wandb.init(project='stargan-weather', entity='tamsyandro', config={
            "d_lr": config.D_LR,  # Both discriminator and generator learning rate
            "g_lr": config.G_LR,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "image_size": config.IMAGE_SIZE,
            "selected_domains": config.SELECTED_DOMAIN,
            "lambda_cls": config.LAMBDA_CLS,
            "lambda_rec": config.LAMBDA_REC,
            "lambda_gp": config.LAMBDA_GP,
            "n_critic": config.N_CRITIC,
            "num_epoch_decay": config.NUM_EPOCHS_DECAY,
            "lr_update_step": config.LR_UPDATE_STEP,
            "d_depth": config.D_DEPTH,
            "g_depth": config.G_SAMPLE_DEPTH,
            "g_bottleneck_depth": config.G_BOTTLENECK_DEPTH,
            "weight_init": config.WEIGHT_INIT,
            "with_attention": False,
            "skip_connection": True,
            # ... Add other hyperparameters here
        })

        # Ensure DEVICE is tracked in WandB
        wandb.config.update({"device": config.DEVICE})


