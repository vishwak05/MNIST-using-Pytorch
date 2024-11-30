
# Project Structure

The main implementation can be found in `main.py`, which contains the model architecture, training loop, and evaluation code.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib

## Installation

Install the required libraries using pip:
```bash
pip install torch torchvision matplotlib
```

## Model Architecture

The CNN architecture (referenced in `main.py`) consists of:

```python
class Model_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, 5, 1, 0)
        self.conv2 = nn.Conv2d(50, 50, 5, 1, 0)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc5(x)
        return x
```

- 2 Convolutional layers
- 2 MaxPooling layers
- 3 Fully connected layers
- ReLU activation functions

## Dataset

The project uses the MNIST dataset, which contains:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images of handwritten digits (0-9)

## Training

The model is trained for 10 epochs using:
- Adam optimizer with a learning rate of 0.001
- CrossEntropy loss function
- Batch size of 64

Training progress includes:
- Per-epoch loss tracking
- Training accuracy monitoring
- Test accuracy evaluation

## Visualization

The training process generates a plot comparing training and test accuracy across epochs using `matplotlib`.

## Usage

To run the training:
```bash
python main.py
```

This will:
- Download the MNIST dataset (if not already present)
- Train the model
- Display training progress
- Show the accuracy plot

## Results

The model's performance can be monitored through:
- Training accuracy per epoch
- Test accuracy per epoch
- Visual comparison plot of training vs test accuracy
