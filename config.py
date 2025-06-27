class CFG:
    # General training parameters
    BATCH_SIZE = 1024

    # Wandb configuration
    WANDB_PROJECT = "VAR"
    WANDB_ENTITY = "andrew_tep"

    # VQ-VAE model parameters
    VQVAE_CE_SCALE = 5.0
    VQVAE_LATENT_DIM = 16
    VQVAE_NUM_EMBEDDINGS = 128
    VQVAE_EPOCHS = 100
    VQVAE_LR = 3e-4

    # PixelCNN model parameters
    PIXELCNN_INPUT_SHAPE = (7, 7)
    PIXELCNN_N_LAYERS = 30
    PIXELCNN_N_FILTERS = 128
    PIXELCNN_KERNEL_SIZE = 5
    PIXELCNN_NUM_EMBEDDINGS = 128
    PIXELCNN_EPOCHS = 100
    PIXELCNN_LR = 2e-5
    PIXELCNN_CFG_SCALE = 6.5
    
    
    LEVELS = [1, 2, 4, 7]
    STAGE_IDS = {1: 0, 2: 1, 4: 2, 7: 3}
    
    CHECKPOINTS_PATH = "output_checkpoints"
