import torch
import torch.nn as nn

class CNNmodel(nn.Module):
    def init(self):
        super(CNNmodel, self).init()
        # 1 Conv2d
        self.cnv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        # 1 Max Pool2d
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 2 Conv2d
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # 2 Max Pool2d
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # 3 Conv2d
        self.cnv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        # 3 Max Pool2d
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # 4 Conv2d
        self.cnv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        # 4 Max Pool2d
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # Activation function
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected layers (adjusted input size based on the output shape)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Adjusted to match the flattened tensor size
        self.fc2 = nn.Linear(1024, 2)  # Binary classification (2 classes)

    def forward(self, x):
        # (1) First layer
        out = self.leakyRelu(self.cnv1(x))
        out = self.maxpool1(out)
        # (2) Second layer
        out = self.leakyRelu(self.cnv2(out))
        out = self.maxpool2(out)
        # (3) Third layer
        out = self.leakyRelu(self.cnv3(out))
        out = self.maxpool3(out)
        # (4) Fourth layer
        out = self.leakyRelu(self.cnv4(out))
        out = self.maxpool4(out)

        # Flatten the output
        out = out.view(out.size(0), -1)  # Flatten to feed into fully connected layers

        # Fully connected layers
        out = self.leakyRelu(self.fc1(out))
        out = self.fc2(out)  # Output layer for classification
        return out