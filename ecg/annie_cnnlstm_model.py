import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        
        # 첫번째 Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=20, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)

        # 두번째 Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=30, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=20, stride=2)
        self.dropout2 = nn.Dropout(p=0.1)

        # 세번째 Convolutional Block
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=10, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=20, stride=2)
        self.dropout3 = nn.Dropout(p=0.1)

        # LSTM Block
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.1)

        # Dense Layers
        self.dense1 = nn.Linear(32, 32)
        self.relu_dense1 = nn.ReLU()
        self.dropout_dense1 = nn.Dropout(p=0.1)

        self.dense2 = nn.Linear(32, 16)
        self.relu_dense2 = nn.ReLU()

        # Output Layer - 이중분류를 위해 출력 유닛을 1로 설정
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        # x: ECG 데이터 (Batch Size, Channels, Length)
        
        # Convolutional Blocks
        x = self.dropout1(self.maxpool1(self.relu1(self.conv1(x))))
        x = self.dropout2(self.maxpool2(self.relu2(self.conv2(x))))
        x = self.dropout3(self.maxpool3(self.relu3(self.conv3(x))))

        # print(x.shape) # (32, 16, 591)

        # LSTM Layer - LSTM은 추가적인 차원을 요구하기 때문에 차원 조정이 필요합니다.
        x = x.permute(0, 2, 1) # (Batch Size, Sequence Length, Features)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout_lstm(x[:, -1, :]) # 마지막 시퀀스의 출력만 사용

        # Dense Layers
        x = self.dropout_dense1(self.relu_dense1(self.dense1(x)))
        x = self.relu_dense2(self.dense2(x))

        # Output Layer - 시그모이드 활성화 함수를 사용하여 0과 1 사이의 값을 출력
        x = torch.sigmoid(self.output(x))
        return x

# 모델 인스턴스화 및 요약
model = CNNLSTMModel()
print(model)
