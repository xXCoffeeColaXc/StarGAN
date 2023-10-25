import torch
import numpy as np
import config
from torchvision.utils import save_image

def denorm():
    pass

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def onehot2label(labels):
    """Convert one-hot vectors to label indices """
    return np.where(labels.numpy() == 1)[1]


# THIS IS SHIT
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    rand_idx = torch.randperm(y.size(0)) # Generate target domain labels randomly.
    label_trg = y[rand_idx]

    #label_trg = label_trg.to(config.DEVICE)

    conditions = ['fog', 'night', 'rain', 'snow', 'daytime']

    print(onehot2label(label_trg.cpu()))

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x, label_trg)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, folder + f"/y_gen_{epoch+1}_{conditions[onehot2label(label_trg.cpu())[-1]]}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch+1}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def build_model():
    pass

def print_model():
    pass

def restore_model():
    pass

def save_model():
    pass

def build_tensorboard():
    pass


