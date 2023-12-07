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
from annie_cnnlstm_model import CNNLSTMModel

import json
import yaml
import random
import wandb



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
    


# YAML 파일 불러오기 및 wandb 초기화
with open('/root/harbud/ecg/annie_hypprm.yaml', 'r') as file:
    hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

wandb.init(project='AFIB Detection(Train)',
           config=hyperparameters
           )
wandb.run.name = 'CNN+LSTM'


# wandb.config에서 하이퍼파라미터 사용
learning_rate = wandb.config.learning_rate
batch_size = wandb.config.batch_size
num_epochs = wandb.config.num_epochs
# hidden_units = wandb.config.hidden_units (예시로, 필요하다면 사용)



# 하이퍼파라미터 출력
print(f"Learning Rate: {learning_rate}")
print(f"Batch Size: {batch_size}")
print(f"Number of Epochs: {num_epochs}")
#print(f"Hidden Units: {hidden_units}")



# 모델을 GPU로 옮기기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMModel().to(device)
# Custom1DCNNWithBatchNormAndDropout

# 손실 함수 및 옵티마이저 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)



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

    epochs_no_improve = 0
    early_stop = False
    patience = 10

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


        # 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early Stopping 체크 및 모델 저장
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping")
                break

        # log metrics to wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_auroc': train_auroc,
            'val_auroc': val_auroc
        })


    # 전체 학습 과정의 결과를 JSON 파일로 저장
    with open('annie_best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {e}")