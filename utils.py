import torch
import numpy as np
import config
import os
from torchvision.utils import save_image

def condition2onehot(labels):
    condition_labels = {}
    # if len(labels) == 2:
    #     condition_labels[labels[0]] = [0]
    #     condition_labels[labels[1]] = [1]
    # else:
    # Generate one-hot encoded vectors
    for i, condition in enumerate(labels):
        one_hot_vector = [0] * len(labels)  # Initialize a list of zeros
        one_hot_vector[i] = 1  # Set the bit for the current condition
        condition_labels[condition] = one_hot_vector  # Assign the vector to the condition
    return condition_labels

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def onehot2label(labels):
    """Convert one-hot vectors to label indices """
    return np.where(labels.numpy() == 1)[1]

def denormalize(tensor, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    denormalized = tensor.clone()
    for t, m, s in zip(denormalized, means, stds):
        t.mul_(s).add_(m)  # Reverse the normalization, original_pixel=normalized_pixel×std+mean
    return denormalized

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    rand_idx = torch.randperm(y.size(0)) # Generate target domain labels randomly.
    label_trg = y[rand_idx]

    if config.ENABLE_DEBUGGING:
        print(onehot2label(label_trg.cpu()))
    target_indices = onehot2label(label_trg.cpu())
    gen.eval()
    with torch.no_grad():
        x_fake = gen(x, label_trg)
        x_fake = x_fake * 0.5 + 0.5  # remove normalization
        x = x * 0.5 + 0.5
        batch_labels = [config.SELECTED_DOMAIN[l] for l in target_indices]
        save_image(x_fake, folder + f"/gen_{epoch}_{batch_labels}.png")
        save_image(x, folder + f"/input_{epoch}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"The directory {checkpoint_file} does not exist.")
    else:

        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def setup_folders():
    project_folders = [
        config.DATA_DIR,
        config.LOG_DIR,
        config.OUTPUT_IMG_DIR,
        config.OUTPUT_MODELS_DIR
    ]

    for folder in project_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Directory {folder} created.")
        else:
            print(f"Directory {folder} already exists.")




