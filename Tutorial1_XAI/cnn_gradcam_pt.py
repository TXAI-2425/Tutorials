import tqdm

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn

import torchvision

from torcheval import metrics

NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_EPOCHS = 5

#### 1. Dataset and DataLoader

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=data_transform
)

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=data_transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

#### 2. Model architecture

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(64*7*7, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = torch.relu(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc(x) # Notice no softmax in PyTorch - it's applied automatically by the cross entropy loss!
        return x

cnn = CNN(num_classes=NUM_CLASSES)

## alternative formulation using nn.Sequential

cnn_sequential = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.5),
    nn.Flatten(),
    nn.Linear(64*7*7, NUM_CLASSES)
)

#### 3. Loss function and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

#### 4. Training loop

cnn.train()
for epoch in range(NUM_EPOCHS):
    accuracy_counter = metrics.MulticlassAccuracy()
    loss_counter = metrics.Mean()
    
    progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    for data, labels in progress_bar:
        optimizer.zero_grad()
        predictions = cnn(data)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        
        loss_counter.update(loss)
        accuracy_counter.update(predictions, labels)
        
        progress_bar.set_postfix(
            loss=loss_counter.compute().item(), 
            accuracy=accuracy_counter.compute().item()
        )

#### 5. Evaluation loop

cnn.eval() # IMPORTANT!
accuracy_counter = metrics.MulticlassAccuracy()
loss_counter = metrics.Mean()

with torch.no_grad():
    progress_bar = tqdm.tqdm(test_loader, desc = "Eval 1/1")
    for data, labels in test_loader:
        predictions = cnn(data)
        loss = loss_fn(predictions, labels)
        
        loss_counter.update(loss)
        accuracy_counter.update(predictions, labels)
        
        progress_bar.set_postfix(
            loss=loss_counter.compute().item(), 
            accuracy=accuracy_counter.compute().item()
        )

#### 6. Save the model
torch.save(cnn.state_dict(), 'cnn.pt')

#### 7. Grad-CAM

# Uncomment to load weights
# cnn.load_state_dict(torch.load('cnn.pt'))

#### 7a. Set up the model for feature extraction - forward hooks

features = []

def forward_hook(module, input_, output):
    features.append(output.detach())

fw_hook = cnn.conv2.register_forward_hook(forward_hook)

#### 7. Set up the model for feature extraction - backward hooks

gradients = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

bw_hook = cnn.conv2.register_backward_hook(backward_hook)

#### 7c. Eval the model

# Now, by passing data to the model, we will be intercepting the features & gradients and appending them to the features list

cnn.eval() # IMPORTANT!
# torch.no_grad() removed since it prevents gradients to be backpropagated, which we need instead

progress_bar = tqdm.tqdm(test_loader, desc = "Eval 1/1")
for data, labels in test_loader:
    predictions = cnn(data)
    predictions.backward(torch.ones_like(predictions))


#### 7d. Grad-CAM

# Create tensors out of features and gradients lists

features = torch.cat(features)
gradients = torch.cat(gradients)

# First, we compute the per-channel weights
weights = gradients.mean(dim=(2, 3), keepdim=True)

# Then, we compute the weighted combination of the features
cam = (weights * features)

# Then we sum the channels to get the final heatmap
cam = cam.sum(dim=1)

# Finally, we apply ReLU to the heatmap
cam = torch.relu(cam)

#### 7e. plotting the heatmap

# Normalize the heatmap - in this case we operate a global normalization
# this means that the heatmap will be normalized across all images
# the original implementation considers a per-image normalization
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Upsample the heatmap to the original image size
# notice: using different interpolation methods can lead to different results
resize = torchvision.transforms.Resize(
    (28, 28),
    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
)
cam = resize(cam)

def plot_heatmap(data, heatmap):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(data.squeeze(), cmap='gray')
    ax[1].imshow(heatmap.squeeze(), cmap='hot', alpha=0.5)
    plt.show()

# Pick a random image from the test set
idx = np.random.randint(0, len(test_set))
data, label = test_set[idx]

# Get the heatmap for the image
heatmap = cam[idx]
plot_heatmap(data, heatmap)
