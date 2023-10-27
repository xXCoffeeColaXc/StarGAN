import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directories
DAWN_DATASET_DIR = "/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/766ygrbt8y-3/DAWN"
ACDC_DATASET_DIR = "/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/rgb_anon_trainvaltest/rgb_anon"
LOG_DIR = "logs/"
RESULT_DIR = "results/"
EVAL_DIR = "eval/"
SAVED_MODELS_DIR = "saved_models/"

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

# Model config
IMAGE_SIZE = 128
CHANNEL_IMG = 3
NUM_DOMAINS = 5
LAMBDA_CLS = 1
LAMBDA_REC = 10
LAMBDA_GP = 10

# Training config
BETA1 = 0.5 # for ADAM 
BETA2 = 0.999 # for ADAM (this is the default, no ?)
LEARNING_RATE = 1e-4 # for both generater and discriminator
BATCH_SIZE = 8
NUM_EPOCHS = 200
N_CRITIC = 5 # number of D updates per each G update 
NUM_WORKERS = 4
SAVE_MODEL = True
LOAD_MODEL = False
LOG_STEP = 10
LOG = True

