import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from tqdm import tqdm
import numpy as np
import seaborn as sns

from a_visualization import plot_roc_and_prc, plot_confusion_matrix
from a_data_loader import create_dataloaders, load_sph_data
from a_models import Custom1DCNN  

def find_optimal_threshold_by_auprc(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = []

    for p, r in zip(precision, recall):
        if p + r == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * p * r / (p + r))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


criterion = nn.BCEWithLogitsLoss()


def test_model(model, test_loader, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loss, test_preds, test_targets = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {data_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs.view(-1), labels.float()).item()
            test_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
            test_targets.extend(labels.cpu().numpy())

    # AUPRC를 기반으로 계산된 최적의 임계값을 얻기 위해 위에서 정의한 함수를 호출합니다.
    optimal_threshold = find_optimal_threshold_by_auprc(test_targets, test_preds)
    test_accuracy = accuracy_score(test_targets, np.array(test_preds) > optimal_threshold)
    test_loss /= len(test_loader)
    test_auroc = roc_auc_score(test_targets, test_preds)
    test_auprc = average_precision_score(test_targets, test_preds)

    print(f'[{data_name} Test] AUROC: {test_auroc:.4f} / AUPRC: {test_auprc:.4f} / Accuracy: {test_accuracy * 100:.2f}% / Loss: {test_loss:.4f}')
    
    plot_roc_and_prc(test_targets, test_preds, data_name)
    plot_confusion_matrix(test_targets, test_preds, optimal_threshold, data_name)


def test_datasets(model_class, model_filename):
    try:
        model = model_class()
        model.load_state_dict(torch.load(model_filename))

        # PTB 데이터셋 테스트
        _, _, ptb_test_loader = create_dataloaders()
        test_model(model, ptb_test_loader, "PTB")

        # SPH 데이터셋 테스트
        sph_test_loader = load_sph_data()
        test_model(model, sph_test_loader, "SPH")

    except Exception as e:
        print(f"Testing failed: {e}")

# 두 데이터셋을 함께 테스트
test_datasets(Custom1DCNN, 'trial_{best_trial.number+1}_best_model.pth')
