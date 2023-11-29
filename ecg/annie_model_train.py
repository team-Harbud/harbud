import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from annie_data_prep import create_dataloaders
from annie_cnn_model import SimpleCNN, Custom1DCNNWithBatchNormAndDropout, Custom1DCNN
import json
import yaml
import random


## 모델 시드 고정 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 사용하는 경우
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# YAML 파일 불러오기
with open('/root/harbud/ecg/annie_hypprm.yaml', 'r') as file:
    hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

# 하이퍼파라미터 설정값 사용
learning_rate = hyperparameters['learning_rate']
batch_size = hyperparameters['batch_size']
num_epochs = hyperparameters['num_epochs']
#hidden_units = hyperparameters['hidden_units']

# 하이퍼파라미터 출력 (선택사항)
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Number of Epochs: {num_epochs}")
#print(f"Hidden Units: {hidden_units}")



# 모델을 GPU로 옮기기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Custom1DCNN().to(device)
# Custom1DCNNWithBatchNormAndDropout

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



def train_model():
    # 학습 및 검증 결과 기록을 위한 사전 선언
    best_model_info = {}

    # 학습 및 검증 과정에서의 손실과 정확도 기록
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_aurocs = []

    train_loader, val_loader, _ = create_dataloaders(batch_size)
    best_auroc = float('-inf')  # 최고 성능 기록을 위한 초기값 설정


    for epoch in range(num_epochs):
        # 훈련 루프
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 그래디언트 초기화

            # 순전파 및 역전파
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(train_targets, np.round(train_preds))
        train_auroc = roc_auc_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)



        # 검증 루프
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels.float())
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_targets, np.round(val_preds))
        val_auroc = roc_auc_score(val_targets, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aurocs.append(val_auroc)



        # 에포크 결과 기록
        best_model_info[epoch+1] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_auroc': train_auroc,
            'val_auroc': val_auroc
        }

        # 최고 성능 모델 업데이트
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), 'model.pth')

    # 전체 학습 과정의 결과를 JSON 파일로 저장
    with open('annie_best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {e}")