import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directories
DAWN_DATASET_DIR = "/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/766ygrbt8y-3/DAWN"
ACDC_DATASET_DIR = "/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/rgb_anon_trainvaltest/rgb_anon"
# ACDC_DATASET_DIR = "/data/rgb_anon"
ROOT_DIR = ''
DATA_DIR = ROOT_DIR + 'data/'
LOG_DIR = ROOT_DIR + "logs/"
RUN_NAME = "BASELINE_2" # 2 condition
OUTPUT_IMG_DIR = ROOT_DIR + "output/images/" + RUN_NAME
OUTPUT_MODELS_DIR = ROOT_DIR + "output/saved_models/" + RUN_NAME
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

# Model config
IMAGE_SIZE = 128
CHANNEL_IMG = 3
SELECTED_DOMAIN = ['daytime', 'fog'] # ['fog', 'night', 'rain', 'snow', 'daytime']
NUM_DOMAINS = len(SELECTED_DOMAIN)
LAMBDA_CLS = 1
LAMBDA_REC = 10
LAMBDA_GP = 10

# Training config
BETA1 = 0.5 # for ADAM 
BETA2 = 0.999 # for ADAM (this is the default, no ?)
D_LR = 1e-4
G_LR = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 200
NUM_EPOCHS_DECAY = 100
LR_UPDATE_STEP = 1000
N_CRITIC = 5 # number of D updates per each G update 
NUM_WORKERS = 4
SAVE_MODEL = False
LOAD_MODEL = False
LOG_STEP = 10
LOG = False
