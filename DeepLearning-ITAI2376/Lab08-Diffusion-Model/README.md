# Lab 08 — Diffusion Model for Image Generation

## Problem Statement
Implement a diffusion-based generative model from scratch to understand how modern generative AI systems create images — from adding noise to training a neural network that reverses the process.

## Approach & Methodology
1. Set up a reproducible environment with fixed random seeds
2. Selected and prepared a dataset (MNIST/Fashion-MNIST/CIFAR-10) using PyTorch DataLoaders
3. Designed a U-Net architecture with:
   - GELUConvBlock for convolution with normalization and activation
   - DownBlock / UpBlock for spatial dimension manipulation with skip connections
   - SinusoidalPositionEmbedBlock for time-step encoding
   - EmbedBlock for class conditioning
4. Implemented forward diffusion (gradual noise addition) and reverse diffusion (learned denoising)
5. Trained the model using MSE loss with gradient clipping and early stopping
6. Generated new images from pure noise and evaluated with CLIP

## Key Concepts
- Diffusion process: forward (noise addition) and reverse (denoising)
- U-Net architecture with skip connections
- Sinusoidal time embeddings
- Noise scheduling (variance schedule)
- CLIP-based semantic evaluation of generated images
- GPU memory management for training and evaluation

## Results & Evaluation
- Successfully generated recognizable images from pure noise
- Visualized the step-by-step denoising process
- Measured image-label alignment using CLIP scores
- Demonstrated class-conditional generation

## Requirements
- Python 3.x
- PyTorch
- CLIP (OpenAI)
- NumPy, Matplotlib
- GPU recommended

## Files
- `MD_Notebook_Banegaserick_ITAI.ipynb` — Completed diffusion model notebook
- `MD_Notebook_Completed_MNIST_ITAI.ipynb` — MNIST completed version
- `L08_Diffusion_student_notebook_.ipynb` — Starter notebook
- `MD_Report_Banegas_Erick_ITAI.pdf` — Written report
