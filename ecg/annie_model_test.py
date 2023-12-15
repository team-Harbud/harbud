import torch
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
from annie_data_prep import create_dataloaders, load_sph_data
from annie_cnn_model import Custom1DCNN
from annie_cnnlstm_model import CNNLSTMModel
from tqdm import tqdm
import wandb



# 특정 임계값에 대한 특이도를 계산하는 함수
def calculate_specificity(y_true, y_pred, thresh):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > thresh).ravel()
    specificity = tn / (tn + fp)
    return specificity


# 최적의 임계값을 찾기 위한 함수
def find_optimal_cutoff(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificity = 1 - fpr
    optimal_idx = np.argmax(specificity >= 0.9)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


# wandb 초기화
wandb.init(project="Test Model Performance")


# 테스트 모델 함수: 모델, 로더, 데이터 이름을 인자로 받음
def test_model(model, test_loader, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {data_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
            test_targets.extend(labels.cpu().numpy())

    optimal_threshold = find_optimal_cutoff(test_targets, test_preds)
    test_accuracy = accuracy_score(test_targets, np.array(test_preds) > optimal_threshold)
    test_loss = np.mean([torch.nn.BCEWithLogitsLoss()(torch.tensor(pred), torch.tensor(target).float()).item() for pred, target in zip(test_preds, test_targets)])
    test_auroc = roc_auc_score(test_targets, test_preds)
    test_auprc = average_precision_score(test_targets, test_preds)


    # wandb를 사용해 Confusion Matrix 생성 및 로깅
    wandb_confmat = wandb.plot.confusion_matrix(
        preds=np.round(np.array(test_preds) > optimal_threshold), 
        y_true=test_targets,
        class_names=["Negative", "Positive"]
    )

    wandb.log({
        f'{data_name} Test Accuracy': test_accuracy,
        f'{data_name} Test Loss': test_loss,
        f'{data_name} Test AUROC': test_auroc,
        f'{data_name} Test AUPRC': test_auprc,
        f'{data_name} Confusion Matrix': wandb_confmat
    })

    print(f'{data_name} Test Accuracy: {test_accuracy * 100:.2f}% - Test AUROC: {test_auroc:.4f} - Test AUPRC: {test_auprc:.4f}')


# PTB 데이터셋을 테스트
def test_model_ptb():
    try:
        model = Custom1DCNN()
        model.load_state_dict(torch.load('model.pth'))
        _, _, test_loader = create_dataloaders()
        test_model(model, test_loader, "PTB")
    except Exception as e:
        print(f"PTB testing failed: {e}")


# SPH 데이터셋을 테스트
def test_model_sph():
    try:
        model = Custom1DCNN()
        model.load_state_dict(torch.load('model.pth'))
        sph_loader = load_sph_data()
        test_model(model, sph_loader, "SPH")
    except Exception as e:
        print(f"SPH testing failed: {e}")

if __name__ == "__main__":
    try:
        test_model_ptb()  # PTB 데이터셋에 대한 테스트
        test_model_sph()  # SPH 데이터셋에 대한 테스트
    except Exception as e:
        print(f"Testing failed: {e}")
