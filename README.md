# Frequency-Aware Noise Injection for Diffusion-Based Image Synthesis

MEng Final Year Project — Imperial College London, 2025
Supervised by Dr C. Ling

---

## Overview

This project extends Denoising Diffusion Probabilistic Models (DDPMs) with a **learnable frequency mask** applied during the forward noise process. Rather than injecting noise uniformly in pixel space, noise is filtered in the frequency domain via `torch.fft`, allowing the model to learn which spectral components to perturb. The reverse process remains identical to standard DDPMs.

## Repository Structure

| Notebook | Description |
|---|---|
| `CIFAR_10_Diffusion_FreqMasking.ipynb` | Main experiment — frequency-aware DDPM with learnable mask on CIFAR-10 |
| `CIFAR_Test_OG.ipynb` | Baseline DDPM on CIFAR-10 (no frequency masking) |
| `MNIST_Diffusion_FreqMasking.ipynb` | Learnable mask validation on MNIST |
| `MNIST_Diffusion_FreqBandHighpass.ipynb` | Fixed high-pass frequency band experiment on MNIST |
| `MNIST_Diffusion_FreqBandLowpass.ipynb` | Fixed low-pass frequency band experiment on MNIST |
| `MNIST_Diffusion_OG_modified.ipynb` | Baseline DDPM on MNIST |

## How to Run

All notebooks are designed for **Google Colab** with a GPU runtime (tested on T4).

1. Open any notebook in Colab
2. Set runtime to GPU (`Runtime > Change runtime type > T4 GPU`)
3. Run all cells from top to bottom

Datasets (CIFAR-10, MNIST) are downloaded automatically via `torchvision`.

## Requirements

```
torch
torchvision
tqdm
matplotlib
clean-fid        # for FID evaluation only
```

No `requirements.txt` is needed for Colab — all packages are either pre-installed or installed inline via `!pip install`.

## Key Implementation

The core contribution is frequency-selective noise injection during the forward process:

```python
# Forward process with learnable frequency mask
x0_freq = torch.fft.fft2(reals, norm='ortho')
eps_freq = torch.fft.fft2(noise, norm='ortho')
mask = torch.sigmoid(mask_param)          # learnable, time-independent
eps_freq = eps_freq * mask
xt = torch.fft.ifft2(x0_freq * alphas_ + eps_freq * sigmas_, norm='ortho').real
```

The mask is jointly optimised with the denoising network using the Adam optimiser.

## License

Based on an original DDPM implementation by Katherine Crowson, licensed under the [MIT License](https://opensource.org/licenses/MIT).
