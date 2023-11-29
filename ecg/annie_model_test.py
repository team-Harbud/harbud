import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from annie_data_prep import create_dataloaders, load_sph_data
from annie_cnn_model import SimpleCNN, Custom1DCNNWithBatchNormAndDropout, Custom1DCNN
from tqdm import tqdm




def test_model_ptb():
    # 모델 및 device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom1DCNN()
    # 저장된 모델 상태 불러오기 
    model.load_state_dict(torch.load('model.pth'))  
    model.to(device)
    model.eval()

    # 테스트 데이터 로더 생성
    _, _, test_loader = create_dataloaders()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_targets, np.round(test_preds))
    test_auroc = roc_auc_score(test_targets, test_preds)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}% - Test AUROC: {test_auroc:.4f}')




def test_model_sph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom1DCNN()
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()

    # SPH 데이터 로더 생성
    sph_loader = load_sph_data()

    sph_preds, sph_targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(sph_loader, desc="Testing SPH"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            sph_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
            sph_targets.extend(labels.cpu().numpy())

    sph_accuracy = accuracy_score(sph_targets, np.round(sph_preds))
    sph_auroc = roc_auc_score(sph_targets, sph_preds)
    print(f'SPH Test Accuracy: {sph_accuracy * 100:.2f}% - SPH Test AUROC: {sph_auroc:.4f}')

if __name__ == "__main__":
    try:
        test_model_ptb()  # PTB 데이터셋에 대한 테스트
        test_model_sph()  # SPH 데이터셋에 대한 테스트
    except Exception as e:
        print(f"Testing failed: {e}")


