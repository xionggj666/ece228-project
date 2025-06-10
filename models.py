import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CNN_2D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), padding=0)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 8 * 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):  # x: (B, 1, 70, 3)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.tanh(self.fc1(x)))
        x = self.dropout(F.tanh(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class CNN_3D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,2,2), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), padding=0)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,2,2), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), padding=0)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3,2,2), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2,2), padding=0)

        self.dropout = nn.Dropout(0.2)


        self.fc1 = nn.Linear(256 * 8 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):  # x: (B, 1, 70, 2, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.tanh(self.fc1(x)))
        x = self.dropout(F.tanh(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class CNN_1D(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, padding=0)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, padding=0)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, padding=0)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 8 , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.tanh(self.fc1(x)))
        x = self.dropout(F.tanh(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x