# Adversarial Training Demo

An interactive Gradio-based demonstration comparing standard neural networks against adversarially-trained robust models under attack. This project uses [RobustBench](https://github.com/RobustBench/robustbench) to showcase how adversarial training improves model resilience against adversarial attacks.

## Overview

This demo implements the **Fast Gradient Sign Method (FGSM)** attack to generate adversarial examples and compares two models:

- **Standard ResNet**: A regular model trained on clean images only
- **Robust ResNet (Carmon2019)**: An adversarially-trained model from the RobustBench leaderboard

The application allows users to upload images, adjust attack strength, and visualize how different models respond to adversarial perturbations.

## Features

- Interactive web interface built with Gradio
- Real-time FGSM attack generation
- Side-by-side comparison of standard vs. robust model predictions
- Visualization of adversarial noise (amplified for visibility)
- Support for custom image uploads (auto-resized to 32x32 for CIFAR-10)

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster inference)
- Docker (optional, for containerized deployment)

### Setup

#### Option 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/congduytran12/Adversarial-Training.git
cd Adversarial-Training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or use the Jupyter notebook to install:
```bash
jupyter notebook app.ipynb
```

#### Option 2: Docker Installation

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

4. Pull and run the container from Docker Hub (Most Recommended):
```bash
docker run -p 7860:7860 congduytran12/adversarial-training-demo
```

The application will be available at `http://localhost:7860`

## Usage

### Running the Demo

**Option 1: Python Script**
```bash
python app.py
```

**Option 2: Jupyter Notebook**
Open `app.ipynb` and run the cells sequentially.

**Option 3: Docker**
```bash
docker run -p 7860:7860 adversarial-training-demo
```

The Gradio interface will launch in your browser (typically at `http://localhost:7860`).

### Using the Interface

1. **Upload an Image**: Click to upload any image (it will be resized to 32x32 pixels)
2. **Adjust Attack Strength**: Use the epsilon slider (0.0 to 0.1) to control perturbation intensity
3. **Run Attack**: Click the "Run Attack" button to generate adversarial example
4. **Compare Results**: View predictions from both standard and robust models

## How It Works

### FGSM Attack

The Fast Gradient Sign Method (FGSM) generates adversarial examples by:

1. Computing the gradient of the loss with respect to the input image
2. Taking the sign of the gradient
3. Adding a small perturbation in that direction:
   ```
   x_adv = x + ε * sign(∇_x J(θ, x, y))
   ```

Where:
- `x` is the original image
- `ε` (epsilon) controls the perturbation magnitude
- `J` is the loss function

### Models

**Standard Model**: Regular ResNet trained on CIFAR-10 clean images
- High accuracy on clean data
- Vulnerable to adversarial perturbations

**Carmon2019Unlabeled Model**: State-of-the-art robust model
- Trained with adversarial examples and unlabeled data
- Maintains accuracy even under attack
- Featured on RobustBench leaderboard

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

Key parameters in `app.py`:

- `DEVICE`: Auto-detects CUDA/CPU
- `epsilon`: Attack strength (0.0 - 0.1)
- Noise amplification: 50x for visualization
- Image size: 32x32 (CIFAR-10 standard)

## Project Structure

```
Adversarial-Training/
├── app.py           # Main Gradio application
├── app.ipynb        # Jupyter notebook version
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

### Attack Parameters

- **Epsilon Range**: 0.00 - 0.10
- **Step Size**: 0.005
- **Threat Model**: L∞ (infinity norm)
- **Target Dataset**: CIFAR-10

## References

- [RobustBench: A standardized adversarial robustness benchmark](https://robustbench.github.io/)
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