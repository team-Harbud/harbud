import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import numpy as np
import random
import json
import logging



# 사용자 정의 모듈 임포트
from a_data_loader import create_dataloaders, load_sph_data
from a_models import Custom1DCNN


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def objective(trial):
    """
    Optuna를 사용하여 하이퍼파라미터 튜닝을 수행하는 함수입니다.
    이 함수는 각 시도(trial)에 대해 학습률, 배치 크기, 에포크 수, 옵티마이저를 선택하고,
    모델을 학습한 후 최적의 AUPRC를 반환합니다.

    Args:
        trial (optuna.trial): Optuna 트라이얼 객체

    Returns:
        float: 해당 트라이얼의 최고 AUPRC 값
    """

    # 모델 시드 고정
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 하이퍼파라미터 제안
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    #num_epochs = trial.suggest_int('num_epochs', 40, 100)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # 모델 선택 및 GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom1DCNN().to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCEWithLogitsLoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    num_epochs = 4

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, verbose=True)
    
    # 데이터 로더 생성
    train_loader, val_loader, _ = create_dataloaders(batch_size)


    try:
        # 학습 및 검증 결과 기록을 위한 사전 선언
        model_info = {}

        # 학습 및 검증 과정에서의 손실과 정확도 기록
        train_losses = []
        train_accuracies = []
        train_aurocs = []  # 훈련 데이터 AUROC 기록을 위한 리스트
        train_auprcs = []  # 훈련 데이터 AUPRC 기록을 위한 리스트
        val_losses = []
        val_accuracies = []
        val_aurocs = []
        val_auprcs = []  # AUPRC 기록을 위한 리스트 추가

        
        #best_auroc = float('-inf')  # 최고 AUROC 기록을 위한 초기값 설정
        best_auprc = float('-inf')  # 최고 AUPRC 기록을 위한 초기값 설정
        best_auprc_info = None  # 최고 AUPRC 값을 가진 모델의 정보를 저장할 변수

        epochs_no_improve = 0
        early_stop = False
        patience = 20

        for epoch in range(num_epochs):
            # 훈련 루프
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []

            for inputs, labels in tqdm(train_loader, desc=f"Trial {trial.number+1} - Epoch {epoch+1}/{num_epochs} - Training"):
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
            train_auprc = average_precision_score(train_targets, train_preds)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_aurocs.append(train_auroc)
            train_auprcs.append(train_auprc)
            print(f">>> [Train] AUROC: {train_auroc:.4f} / AUPRC: {train_auprc:.4f}")



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
            val_auprc = average_precision_score(val_targets, val_preds)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_aurocs.append(val_auroc)
            val_auprcs.append(val_auprc)

            print(f">>> [Valid] AUROC: {val_auroc:.4f} / AUPRC: {val_auprc:.4f}")


            # 에포크 결과 기록
            epoch_info = {
                'train_loss': train_loss,
                'valid_loss': val_loss,
                'train_accuracy': train_accuracy,  
                'valid_accuracy': val_accuracy,      
                'train_auroc': train_auroc,
                'valid_auroc': val_auroc,
                'train_auprc': train_auprc,        
                'valid_auprc': val_auprc
            }
            model_info[epoch + 1] = epoch_info


            # 스케줄러 업데이트
            scheduler.step(val_auprc)

            # Early Stopping 체크 및 모델 저장
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                epochs_no_improve = 0
                best_auprc_info = epoch_info  # 최고 AUPRC 값을 갱신할 때 정보 저장
                # 최고 성능 모델 저장
                torch.save(model.state_dict(), f'trial_{trial.number+1}_best_model.pth')
                
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping")
                    break
        

            # 전체 학습 과정의 결과를 JSON 파일로 저장
            with open(f'trial_{trial.number+1}_performance.json', 'w') as f:
                json.dump(model_info, f, indent=4)




        # 최고 AUPRC 값을 가진 모델의 정보 출력
        logging.info("-" * 42)
        logging.info(f"< Trial {trial.number+1}'s Best Performance>")
        if best_auprc_info is not None:
            items = list(best_auprc_info.items())
            for i, (key, value) in enumerate(items):
                message = f"[{key}]: {value:.4f}"
                if i == len(items) - 1:
                    message += " <- Pick It Up!"
                logging.info(message)


    except Exception as e:
        logging.error(f"학습 중 에러 발생: {e}")
        raise e   


    logging.info(f"최고 AUPRC: {best_auprc:.4f} - Trial {trial.number+1}")

    return best_auprc


