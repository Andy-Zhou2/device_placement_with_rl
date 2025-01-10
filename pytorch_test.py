import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define the model with layers on different devices
class TwoDeviceModel(nn.Module):
    def __init__(self):
        super(TwoDeviceModel, self).__init__()
        self.layer1 = nn.Linear(10, 50).to('cuda')  # Layer 1 on GPU
        self.layer2 = nn.Linear(50, 1).to('cpu')    # Layer 2 on CPU

    def forward(self, x):
        x = self.layer1(x.to('cuda'))  # Forward pass on GPU
        x = x.to('cpu')               # Move data to CPU
        x = self.layer2(x)            # Forward pass on CPU
        return x

# Create the artificial dataset
def create_dataset(num_samples=1000):
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    return X, y

# Initialize model, dataset, and optimizer
model = TwoDeviceModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
X, y = create_dataset()

# Measure time for a forward and backward pass
start_time = time.time()

# Forward pass
outputs = model(X)
loss = criterion(outputs, y)

# Backward pass
loss.backward()

# Optimizer step
optimizer.step()

end_time = time.time()

print(f"Time for one forward and backward pass: {end_time - start_time:.4f} seconds")
