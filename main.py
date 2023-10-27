from torchvision import transforms
from data_loader import ACDCDataset
from torch.utils.data import DataLoader
import config
from stargan import StarGAN

def main():
    # Create Datalaoder
    transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(config.IMAGE_SIZE), # is this doing anything ?
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)


    val_dataset = ACDCDataset(root_dir=config.ACDC_DATASET_DIR, transform=transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


    # Initialize StarGAN
    model = StarGAN(train_loader=train_loader, val_loader=val_loader)

    model.print_model()

    model.train()


    '''
    # Initialize generator and discriminator
    disc = Discriminator(image_size=config.IMAGE_SIZE, in_channels=config.CHANNEL_IMG, features=64, c_dim=config.NUM_DOMAINS)
    disc = disc.to(config.DEVICE)

    gen = Generator(in_channels=config.CHANNEL_IMG, feautues=64, c_dim=config.NUM_DOMAINS)
    gen = gen.to(config.DEVICE)

   

    # Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    criterion_cycle = torch.nn.L1Loss()
    criterion_cycle = criterion_cycle.to(config.DEVICE)

    
    if config.LOAD_MODEL:
        load_checkpoint(os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_GEN), gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_DISC), disc, opt_disc, config.LEARNING_RATE)
    else:
        gen.apply(weights_init_normal)
        disc.apply(weights_init_normal)

    # Start training.
    print('Start training...')
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc=disc, gen=gen, loader=train_loader, g_opt=opt_gen, d_opt=opt_disc, start_time=start_time) # could pass GradientScaler

        if config.SAVE_MODEL and epoch%5==0:
            pass
            #TODO save_checkpoint(gen, opt_gen, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_GEN))
            #TODO save_checkpoint(disc, opt_disc, filename=os.path.join(config.SAVED_MODELS_DIR, config.CHECKPOINT_DISC))

        # Save images for debugging
        save_some_examples(gen, val_loader, epoch, folder=config.EVAL_DIR)
    '''

if __name__ == "__main__":
    main()
    




    