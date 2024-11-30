import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Transform code
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Build the architecture
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

model = Model_0()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# To store accuracy values
train_accuracy = []
test_accuracy = []

# Training loop
for epoch in range(10):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    model.train()
    for image, label in train_loader:
        image = image.view(image.shape[0], 1, 28, 28)
        prediction = model(image)
        loss = criterion(prediction, label)  # loss
        loss.backward()  # backward
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # Track training accuracy
        _, predicted_train = torch.max(prediction.data, 1)
        total_train += label.size(0)
        correct_train += (predicted_train == label).sum().item()

    # Calculate and store training accuracy
    train_acc = 100 * correct_train / total_train
    train_accuracy.append(train_acc)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_acc}%")

    # Evaluate on the test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.view(image.shape[0], 1, 28, 28)
            predictions = model(image)
            _, predicted_test = torch.max(predictions.data, 1)
            total_test += label.size(0)
            correct_test += (predicted_test == label).sum().item()

    # Calculate and store test accuracy
    test_acc = 100 * correct_test / total_test
    test_accuracy.append(test_acc)
    print(f"Test Accuracy after Epoch {epoch+1}: {test_acc}%")

# Plot accuracy
epochs = range(1, 11)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()
