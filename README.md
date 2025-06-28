# miniVAR: VQ-VAE + PixelCNN

This repository is a lightweight, teaching-scale re-implementation of the main idea from [**Visual Autoregressive Modeling (VAR)**](https://arxiv.org/abs/2404.02905): a coarse-to-fine, next-scale prediction pipeline. Instead of heavy DiT transformers, we plug in a simple PixelCNN with a small “stage” embedding to condition on previous scales.


<!-- ## Features
- **VQ-VAE**: Discrete latent representation learning for images
- **PixelCNN**: Powerful autoregressive prior over VQ-VAE latents
- **Multi-scale training**: Support for hierarchical latent structures
- **WandB integration**: Track experiments and visualize losses
- **Visualization tools**: Sample generation, reconstructions, and training curves
- **Configurable**: All hyperparameters and model settings in `config.py` -->

## ✅ What’s Done
- [x] **VectorQuantizer** with classic straight-through loss + small MLPs φ_i for multi-scale canvas accumulation.
- [x] **VQ-VAE:** convolutional encoder/decoder for MNIST (28×28→7×7 latents).
- [x] Lightweight **PixelCNN** with masked convolutions (A → B), purely unconditional.
- [x] Conditional “stage” embedding that tells the model which up-sampling step it’s at (1→2, 2→4, 4→7).
- [x] At each scale, we upscale previous indices, one-hot, add stage embedding, predict next indices. (Next-Scale Prediction)
- [x] **WandB** integration: track experiments and visualize losses

## 🔧 What Can Be Improved (for future)
- [ ] Replace PixelCNN with a small Transformer or DiT block for richer inductive bias.
- [ ] Try on CIFAR-10 or a small face dataset to see generative power beyond MNIST.
- [ ] Experiment with more scales (e.g. 1→3→7→14→28).
 
## Project Structure
```
miniVAR/
├── models/                # Model definitions (VQ-VAE, PixelCNN, VectorQuantizer)
├── utils/                 # Training loops, visualization, and helpers
├── train_vq_vae.py        # Script to train VQ-VAE
├── train_pixel_cnn.py     # Script to train PixelCNN prior
├── config.py              # Centralized configuration
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── myminivar.ipynb        # (Optional) Jupyter notebook for experiments
```

## Installation
1. Clone the repository and navigate to the project folder:
   ```sh
   git clone https://github.com/teplov-andrew/miniVAR.git
   cd miniVAR
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### 1. Train VQ-VAE
```sh
python train_vq_vae.py
```
- Trains a VQ-VAE model on MNIST.
- Saves model weights and visualizations.

### 2. Train PixelCNN Prior
```sh
python train_pixel_cnn.py
```
- Trains a PixelCNN prior on the VQ-VAE latents.
- Generates and saves various sample visualizations.

### 3. Configuration
Edit `config.py` to change model hyperparameters, training settings, or experiment details.

### 4. Visualization
- Visualizations (samples, reconstructions, training curves) are saved as PNG files during/after training.
- Use the provided Jupyter notebook for interactive exploration.


## Conclusion 
The goal was to try to implement the modern algorithm from the article yourself.  

In total, we managed to implement the basic techniques and tricks described in the [VAR](https://arxiv.org/abs/2404.02905) article. The generation quality is not ideal, for this you need to change the models themselves to larger ones. I did this work completely alone, so there may be bugs and flaws in the code. If there are any, I will be glad if you let me know about them.