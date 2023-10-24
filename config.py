import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/766ygrbt8y-3/DAWN"
VAL_DIR = "data/cityscapes/val"

IMAGE_SIZE = 128
CHANNEL_IMG = 3
NUM_DOMAINS = 4

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_EPOCHS = 200

NUM_WORKERS = 4
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
