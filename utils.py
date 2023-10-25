import torch
import numpy as np


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def onehot2label(labels):
    """Convert one-hot vectors to label indices """
    return np.where(labels.numpy() == 1)[1]