import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2) 
        self.pool = nn.MaxPool1d(kernel_size=5, stride=1) 
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5) 
        self.fc1 = nn.Linear(64 * 2486, 1000) 
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
        x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
        x = x.view(-1, 64 * 2486)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


## 충원님께서 돌려보신 모델에서 리키렐루 등을 바꿔보았다. 

class Custom1DCNNWithBatchNormAndDropout(nn.Module):
    def __init__(self):
        super(Custom1DCNNWithBatchNormAndDropout, self).__init__()

        # Convolutional Blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-05)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.maxpool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=5)
        self.maxpool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(64, 128, kernel_size=5)
        self.maxpool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=5)
        self.maxpool6 = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(0.4)

        self.conv7 = nn.Conv1d(128, 256, kernel_size=5)
        self.maxpool7 = nn.MaxPool1d(2)

        self.conv8 = nn.Conv1d(256, 256, kernel_size=5)
        self.maxpool8 = nn.MaxPool1d(2)

        self.dropout2 = nn.Dropout(0.4)

        self.conv9 = nn.Conv1d(256, 512, kernel_size=5)
        self.maxpool9 = nn.MaxPool1d(2)

        self.dropout3 = nn.Dropout(0.4)

        self.conv10 = nn.Conv1d(512, 512, kernel_size=5)
        self.leaky_relu10 = nn.LeakyReLU(negative_slope=0.01)



        # Fully Connected Blocks
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(512, 128)
        self.batch_norm_dense1 = nn.BatchNorm1d(128, eps=1e-05)  # BatchNorm1d for Dense1
        self.dropout4 = nn.Dropout(0.4)

        self.dense2 = nn.Linear(128, 32)
        self.batch_norm_dense2 = nn.BatchNorm1d(32, eps=1e-05)  # BatchNorm1d for Dense2

        self.dense3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()  # 이진 분류를 위한 시그모이드 활성화 함수

    def forward(self, x):
        # Convolutional Blocks
        x = self.maxpool1(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.leaky_relu(self.conv2(x)))
        x = self.maxpool3(self.leaky_relu(self.conv3(x)))
        x = self.maxpool4(self.leaky_relu(self.conv4(x)))
        x = self.maxpool5(self.leaky_relu(self.conv5(x)))
        x = self.maxpool6(self.leaky_relu(self.conv6(x)))
        x = self.dropout1(x)
        x = self.maxpool7(self.leaky_relu(self.conv7(x)))
        x = self.maxpool8(self.leaky_relu(self.conv8(x)))
        x = self.dropout2(x)
        x = self.maxpool9(self.leaky_relu(self.conv9(x)))
        x = self.dropout3(x)
        x = self.leaky_relu10(self.conv10(x))  # 수정

        # Fully Connected Blocks
        x = self.flatten(x)
        x = self.dropout4(self.leaky_relu(self.batch_norm_dense1(self.dense1(x))))
        x = self.leaky_relu(self.batch_norm_dense2(self.dense2(x)))
        x = self.sigmoid(self.dense3(x))  # 수정

        return x


# 충원님 코드

class Custom1DCNN(nn.Module):
    def __init__(self):
        super(Custom1DCNN, self).__init__()

        # Convolutional Blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.maxpool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=5)
        self.maxpool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(64, 128, kernel_size=5)
        self.maxpool5 = nn.MaxPool1d(2)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=5)
        self.maxpool6 = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(0.5)

        self.conv7 = nn.Conv1d(128, 256, kernel_size=5)
        self.maxpool7 = nn.MaxPool1d(2)

        self.conv8 = nn.Conv1d(256, 256, kernel_size=5)
        self.maxpool8 = nn.MaxPool1d(2)

        self.dropout2 = nn.Dropout(0.5)

        self.conv9 = nn.Conv1d(256, 512, kernel_size=5)
        self.maxpool9 = nn.MaxPool1d(2)

        self.dropout3 = nn.Dropout(0.5)

        self.conv10 = nn.Conv1d(512, 512, kernel_size=5)

        # Fully Connected Blocks
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(512, 128)
        self.batch_norm_dense1 = nn.BatchNorm1d(128)  # BatchNorm1d for Dense1
        self.dropout4 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(128, 32)
        self.batch_norm_dense2 = nn.BatchNorm1d(32)  # BatchNorm1d for Dense2

        self.dense3 = nn.Linear(32, 1)
        

    def forward(self, x):
        # Convolutional Blocks
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.maxpool3(self.relu(self.conv3(x)))
        x = self.maxpool4(self.relu(self.conv4(x)))
        x = self.maxpool5(self.relu(self.conv5(x)))
        x = self.maxpool6(self.relu(self.conv6(x)))
        x = self.dropout1(x)
        x = self.maxpool7(self.relu(self.conv7(x)))
        x = self.maxpool8(self.relu(self.conv8(x)))
        x = self.dropout2(x)
        x = self.maxpool9(self.relu(self.conv9(x)))
        x = self.dropout3(x)
        x = self.conv10(x)

        # Fully Connected Blocks
        x = self.flatten(x)
        x = self.dropout4(self.relu(self.batch_norm_dense1(self.dense1(x))))
        x = self.relu(self.batch_norm_dense2(self.dense2(x)))
        x = self.dense3(x)

        return x