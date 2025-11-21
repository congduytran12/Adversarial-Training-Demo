# Adversarial Training Demo

An interactive Gradio-based demonstration comparing standard neural networks against adversarially-trained robust models under attack. This project uses [RobustBench](https://github.com/RobustBench/robustbench) to showcase how adversarial training improves model resilience against adversarial attacks.

## Overview

This demo implements multiple adversarial attack methods including **FGSM** and **AutoAttack** to generate adversarial examples. Users can compare a standard model against various state-of-the-art robust models:

- **Standard ResNet**: A regular model trained on clean images only
- **Robust Models**: Multiple adversarially-trained models from the RobustBench leaderboard:
  - Bartoldson2024 WRN-94-16
  - Amini2024 MeanSparse WRN-94-16
  - Bartoldson2024 WRN-82-8

The application allows users to upload images, select attack methods, adjust attack strength, and visualize how different models respond to adversarial perturbations.

## Features

- Interactive web interface built with Gradio
- Multiple attack methods: FGSM, APGD-CE, APGD-DLR, FAB, Square, and targeted variants
- Multiple robust model choices from RobustBench leaderboard
- Side-by-side comparison of standard vs. robust model predictions
- Visualization of adversarial noise (amplified for visibility)
- Support for custom image uploads (auto-resized to 32x32 for CIFAR-10)
- Docker Hub deployment for easy access

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster inference)
- Docker (optional, for containerized deployment)

### Setup

#### Option 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/congduytran12/Adversarial-Training-Demo.git
cd Adversarial-Training-Demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Option 2: Docker Installation (Recommended)

**Quick Start - Pull from Docker Hub:**
```bash
docker pull congduytran12/adversarial-training-demo
docker run -p 7860:7860 congduytran12/adversarial-training-demo
```

**Or Build Locally:**

1. Clone the repository:
```bash
git clone https://github.com/congduytran12/Adversarial-Training.git
cd Adversarial-Training
```

2. Build the Docker image:
```bash
docker build -t adversarial-training-demo .
```

3. Run the container:
```bash
docker run -p 7860:7860 adversarial-training-demo
```

The application will be available at `http://localhost:7860`

## Usage

### Running the Demo

**Option 1: Python Script**
```bash
python app.py
```

**Option 2: Docker**
```bash
docker run -p 7860:7860 adversarial-training-demo
```

The Gradio interface will launch in your browser (typically at `http://localhost:7860`).

### Using the Interface

1. **Upload an Image**: Upload an image from CIFAR-10 categories (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
2. **Select Robust Model**: Choose from multiple state-of-the-art robust models
3. **Select Attack Method**: Choose between FGSM (fast) or AutoAttack variants (stronger but slower)
4. **Adjust Attack Strength**: Use the epsilon slider (0.0 to 0.1) to control perturbation intensity
5. **Run Attack**: Click the button to generate adversarial example
6. **Compare Results**: View upscaled visualizations and predictions from both standard and robust models

## How It Works

### Attack Methods

**FGSM (Fast Gradient Sign Method)**
A single-step attack that generates adversarial examples by:
1. Computing the gradient of the loss with respect to the input image
2. Taking the sign of the gradient
3. Adding a small perturbation: `x_adv = x + ε * sign(∇_x J(θ, x, y))`

**AutoAttack Methods**
- **APGD-CE/DLR**: Strong iterative gradient-based attacks
- **FAB**: Fast Adaptive Boundary attack for minimal perturbations
- **Square**: Black-box attack using random search
- **APGD-T/FAB-T**: Targeted versions of the above attacks

Where:
- `x` is the original image
- `ε` (epsilon) controls the perturbation magnitude
- `J` is the loss function

### Models

**Standard Model**: Regular ResNet trained on CIFAR-10 clean images
- High accuracy on clean data
- Vulnerable to adversarial perturbations

**Robust Models** (RobustBench leaderboard):
- **Bartoldson2024 WRN-94-16**: Wide ResNet with 94 layers, state-of-the-art defense
- **Amini2024 MeanSparse**: Sparse neural network approach
- **Bartoldson2024 WRN-82-8**: Lighter wide ResNet variant

All robust models maintain accuracy even under strong attacks.

## CIFAR-10 Classes

The models classify images into 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Configuration

Key parameters:

- `DEVICE`: Auto-detects CUDA/CPU
- `epsilon`: Attack strength (0.0 - 0.1, default 0.00)
- Noise amplification: 50x for visualization
- Image size: 32×32 (CIFAR-10 standard)
- Attack methods: FGSM, APGD-CE, APGD-DLR, FAB, Square, APGD-T, FAB-T

## Project Structure

```
Adversarial-Training/
├── app.py           # Main Gradio application
├── requirements.txt # Dependencies
├── Dockerfile       # Docker configuration
├── .dockerignore    # Docker ignore file
├── models/          # Pre-trained model weights
└── README.md        # This file
```

## Technical Details

### Dependencies

- `gradio`: Web interface
- `torch`: Deep learning framework
- `torchvision`: Image transformations
- `robustbench`: Pre-trained robust models
- `autoattack`: Advanced adversarial attack library

### Attack Parameters

- **Epsilon Range**: 0.00 - 0.10
- **Default Epsilon**: 0.00 (standard for CIFAR-10)
- **Step Size**: 0.001
- **Threat Model**: Linf (infinity norm)
- **Target Dataset**: CIFAR-10
- **Attack Speed**: FGSM (<1s), AutoAttack variants (2-30s)

## Docker Hub

The application is available on Docker Hub:
```bash
docker pull congduytran12/adversarial-training-demo
```

[View on Docker Hub](https://hub.docker.com/r/congduytran12/adversarial-training-demo)

## References

- [RobustBench: A standardized adversarial robustness benchmark](https://robustbench.github.io/)
- [AutoAttack: Reliable adversarial robustness evaluation](https://github.com/fra31/auto-attack)
- [Carmon et al., 2019 - Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/abs/1905.13736)
- [Goodfellow et al., 2014 - Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## Acknowledgments

- RobustBench team for the robust model repository
- Gradio for the interactive interface framework
- PyTorch community for the deep learning tools

---

**Note**: The first run will download pre-trained models (~100MB each), which may take a few minutes depending on your internet connection.
