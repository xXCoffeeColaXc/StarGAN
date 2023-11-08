import torch
import time
import config
from tqdm import tqdm
import torch.nn.functional as F
import datetime
import wandb

def update_lr(g_opt, d_opt, g_lr, d_lr):
    """Decay learning rates of the generator and discriminator."""
    for param_group in g_opt.param_groups:
        param_group['lr'] = g_lr
    for param_group in d_opt.param_groups:
        param_group['lr'] = d_lr

    wandb.config.update({"d_lr": d_lr, "g_lr": g_lr}, allow_val_change=True)

def classification_loss(logit, target): 
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


def train_fn(disc, gen, loader, g_opt, d_opt, start_time, epoch):

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
        d_loss_real = - torch.mean(out_src) # 2x2 patch, maximize the scores
        d_loss_cls = classification_loss(out_cls, label_org)

        # Compute loss with fake images.
        x_fake = gen(x_real, c_trg)
        out_src, out_cls = disc(x_fake.detach())
        d_loss_fake = torch.mean(out_src) # 2x2 patch, minimize the scores

        # NOTE: In the WGAN formulation, the discriminator (or critic) tries to maximize the difference between the scores of real and fake images.

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
        # NOTE (d_loss_real + d_loss_fake) / 2 ??
        # NOTE: L(D) = -L(adv) + lamda_cls * L_r(cls)
        d_loss_adversial = (d_loss_real + d_loss_fake)
        d_loss = (d_loss_real + d_loss_fake) + config.LAMBDA_CLS * d_loss_cls + config.LAMBDA_GP * d_loss_gp
        
        reset_grad(g_opt, d_opt) 
        d_loss.backward()
        d_opt.step()

        # Logging
        loss = {}
        loss['D/d_loss_adversial'] = d_loss_adversial.item()
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_cls'] = d_loss_cls.item()
        loss['D/loss_gp'] = d_loss_gp.item()

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        # NOTE: TRY OUT N_CRITIC=5: train generator on every fifth iteration (idx+1 % N_CRITIC == 0), disc learns too fast...

        # Every n_critic times update generator
        if (idx+1) % config.N_CRITIC == 0:

            # Originial-to-target domain
            x_fake = gen(x_real, c_trg)
            out_src, out_cls = disc(x_fake)
            g_loss_fake = - torch.mean(out_src) # Adversarial loss
            g_loss_cls = classification_loss(out_cls, label_trg)  # Classification loss

            # Target-to-original domain
            x_reconst = gen(x_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real-x_reconst)) # NOTE L1 Loss could be criterion_cycle(x_reconst, x_real)

            # Backward and optimize
            # NOTE: L(G) = L(adv) + lamda_cls * L_f(cls) + lambda_rec * L(rec)
            g_loss = g_loss_fake + config.LAMBDA_REC * g_loss_rec + config.LAMBDA_CLS * g_loss_cls
            
            reset_grad(g_opt, d_opt) 
            g_loss.backward()
            g_opt.step()

            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/loss_cls'] = g_loss_cls.item()

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #

        
        # Logging
        # Print out training information.
        if (idx+1) % config.LOG_STEP == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, idx+1, len(loader))
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

        if config.ENABLE_LOGGING:
            wandb.log(loss)


            # LR decay
            # TODO try not to save lr in wandb
            d_lr = wandb.config.d_lr
            g_lr = wandb.config.g_lr

            # Decay learning rates.
            if len(loader) < config.LR_UPDATE_STEP:
                lr_update_step = len(loader) - 1
            else:
                lr_update_step = config.LR_UPDATE_STEP
            if (idx+1) % lr_update_step == 0 and epoch > config.NUM_EPOCHS_DECAY:
                g_lr -= (config.G_LR / float(config.NUM_EPOCHS_DECAY * 100))
                d_lr -= (config.D_LR / float(config.NUM_EPOCHS_DECAY * 100))
                update_lr(g_opt=g_opt, d_opt=d_opt, g_lr=g_lr, d_lr=d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        
            wandb.log({"d_lr": d_lr, "g_lr": g_lr})




def test_fn():
    pass