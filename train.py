import torch
import time
import config
from tqdm import tqdm
import torch.nn.functional as F


def update_lr():
    pass

def test_fn():
    pass

def classification_loss(logit, target): # pass it as a parameter ?
    """Compute binary or softmax cross entropy loss."""
    target = target.float()
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0) 
    # The with_logits implies that it expects raw, unnormalized values (logits) rather than probabilities 
    # (i.e., it internally applies the sigmoid function to the logits).

def gradient_penalty(y,x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(config.DEVICE)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def reset_grad(g_opt, d_opt):
    g_opt.zero_grad()
    d_opt.zero_grad()


def train_fn(disc, gen, loader, g_opt, d_opt):
   loop = tqdm(loader, leave=True)

   for idx, (x,y) in enumerate(loop): # y=tensor([0,0,0,0,1])
       
    # =================================================================================== #
    #                             1. Preprocess input data                                #
    # =================================================================================== #

    label_org = y
    rand_idx = torch.randperm(label_org.size(0)) # Generate target domain labels randomly.
    label_trg = label_org[rand_idx]

    c_org = label_org.clone() # if using ACDC dataset 
    c_trg = label_trg.clone()

    x_real = x.to(config.DEVICE)                # Input images.
    c_org = c_org.to(config.DEVICE)             # Original domain labels.
    c_trg = c_trg.to(config.DEVICE)             # Target domain labels.
    label_org = label_org.to(config.DEVICE)     # Labels for computing classification loss.
    label_trg = label_trg.to(config.DEVICE)     # Labels for computing classification loss.


    # =================================================================================== #
    #                             2. Train the discriminator                              #
    # =================================================================================== #

    # Compute loss with real images
    out_src, out_cls = disc(x_real)
    d_loss_real = - torch.mean(out_src) # 2x2 patch, computes the real/fake loss. The negative sign suggests that the discriminator aims to maximize the score for real images.

    # DONT WE NEED BCE ?? 
    # D_real_loss = bce(D_real, torch.ones_like(D_real)) # example: bce([0.9, 0.8, 0.7], [1, 1, 1])
    # D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
    d_loss_cls = classification_loss(out_cls, label_org)

    # Compute loss with fake images.
    x_fake = gen(x_real, c_trg)
    out_src, out_cls = disc(x_fake.detach())
    d_loss_fake = torch.mean(out_src) # 2x2 patch but not negative ??

    # Compute loss for gradient penalty.
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(config.DEVICE)
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src, _ = disc(x_hat)
    d_loss_gp = gradient_penalty(out_src, x_hat)

    '''
    # WE COULD USE GRADIENT SCALER
    disc.zero_grad() # Gradients of the discriminator parameters are set to zero.
    d_scaler.scale(d_loss).backward() # computes the gradient of the loss with respect to model parameters.
    d_scaler.step(d_opt) # The optimizer opt_disc updates the discriminator weights.
    d_scaler.update() # Updates the gradient scaler.
    '''

    # Backward and optimize
    # d_loss_fake is for generator no ?? || (d_loss_real + d_loss_fake) / 2 ??
    # L(D) = -L(adv) + lamda_cls * L_r(cls)
    d_loss = d_loss_real + d_loss_fake + config.LAMBDA_CLS * d_loss_cls + config.LAMBDA_GP * d_loss_gp
    reset_grad(g_opt, d_opt) # d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()


    # =================================================================================== #
    #                               3. Train the generator                                #
    # =================================================================================== #

    # TRY OUT N_CRITIC=5: train generator on every fifth iteration (idx+1 % N_CRITIC == 0)
    
    # Originial-to-target domain
    x_fake = gen(x_real, c_trg)
    out_src, out_cls = disc(x_fake)
    g_loss_fake = - torch.mean(out_src) # multiplied with -1 idk? AGAIN BCE ????
    g_loss_cls = classification_loss(out_cls, label_trg)

    # Target-to-original domain
    x_reconst = gen(x_fake, c_org)
    g_loss_rec = torch.mean(torch.abs(x_real-x_reconst))

    # Backward and optimize
    # L(G) = L(adv) + lamda_cls * L_f(cls) + lambda_rec * L(rec)
    g_loss = g_loss_fake + config.LAMBDA_REC * g_loss_rec + config.LAMBDA_CLS * g_loss_cls
    reset_grad(g_opt, d_opt) # g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()


    # =================================================================================== #
    #                                 4. Miscellaneous                                    #
    # =================================================================================== #

    # Logging

    # LR decay




