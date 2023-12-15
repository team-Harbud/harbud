import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import json
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns



def plot_prediction_comparison(true_labels, predicted_labels, subplot_position):
    """
    주어진 서브플롯 위치에 실제 라벨과 예측 라벨을 비교하는 그래프를 그립니다.
    
    Args:
        true_labels (list): 실제 라벨의 리스트.
        predicted_labels (list): 예측된 라벨의 리스트.
        subplot_position (int): 서브플롯의 위치 (예: 221, 222 등).
    """
    plt.subplot(subplot_position)
    plt.plot(true_labels, label='실제 라벨', lw=1)
    plt.plot(predicted_labels, label='예측 라벨', linestyle='--')
    plt.title('실제 라벨 vs 예측 라벨', fontsize=12)
    plt.xlabel('Sample', fontsize=8)
    plt.ylabel('Label', fontsize=8)
    plt.legend()

def plot_metric_subplot(train_metric, val_metric, metric_name, subplot_position):
    """
    주어진 서브플롯 위치에 훈련 및 검증 데이터셋에 대한 성능 지표를 그래프로 그립니다.
    
    Args:
        train_metric (list): 훈련 데이터의 성능 지표 리스트.
        val_metric (list): 검증 데이터의 성능 지표 리스트.
        metric_name (str): 그래프에 표시할 성능 지표의 이름.
        subplot_position (int): 서브플롯의 위치 (예: 221, 222 등).
    """
    plt.subplot(subplot_position)
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'b-', label='Train', lw=1)
    plt.plot(epochs, val_metric, 'r-', label='Valid', lw=1)
    plt.title(metric_name, fontsize=12)
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel(metric_name, fontsize=8)
    ax = plt.gca()
    y_vals = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(y_vals))
    ax.set_yticklabels(['{:.2f}'.format(y) for y in y_vals])
    plt.legend()



def load_metrics(filename):
    """
    JSON 파일에서 학습 과정 중의 성능 지표를 로드합니다.
    
    Args:
        filename (str): 성능 지표가 저장된 JSON 파일의 경로.
    
    Returns:
        tuple: 각 성능 지표들의 리스트 (train_losses, val_losses, train_accuracies, val_accuracies, train_aurocs, val_aurocs, train_auprcs, val_auprcs).
    """
    with open(filename, 'r') as file:
        data = json.load(file)
        train_losses = [epoch_info['train_loss'] for epoch_info in data.values()]
        val_losses = [epoch_info['valid_loss'] for epoch_info in data.values()]
        train_accuracies = [epoch_info['train_accuracy'] for epoch_info in data.values()]
        val_accuracies = [epoch_info['valid_accuracy'] for epoch_info in data.values()]
        train_aurocs = [epoch_info['train_auroc'] for epoch_info in data.values()]
        val_aurocs = [epoch_info['valid_auroc'] for epoch_info in data.values()]
        train_auprcs = [epoch_info['train_auprc'] for epoch_info in data.values()]
        val_auprcs = [epoch_info['valid_auprc'] for epoch_info in data.values()]
        return train_losses, val_losses, train_accuracies, val_accuracies, train_aurocs, val_aurocs, train_auprcs, val_auprcs



def plot_roc_and_prc(y_true, y_pred, data_name):
    # ROC 커브
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {data_name}')
    plt.legend(loc="lower right")

    # PRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {data_name}')
    plt.legend(loc="lower left")
    plt.show()



# Confusion Matrix 시각화 및 정밀도, 재현율, F1 점수 출력
def plot_confusion_matrix(y_true, y_pred, thresh, data_name):
    cm = confusion_matrix(y_true, y_pred > thresh)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {data_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 혼동 행렬로부터 TP, FP, FN, TN 추출
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    # 정밀도, 재현율, F1 점수 계산
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # 계산된 지표 출력
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1 점수 (F1 Score): {f1_score:.4f}")




# # 이 부분은 실제 실행 스크립트에서 사용할 수 있습니다.
# if __name__ == "__main__":
#     train_losses, val_losses, train_accuracies, val_accuracies, train_aurocs, val_aurocs, train_auprcs, val_auprcs = load_metrics('f'trial_{trial.number+1}_performance.json'')

#     plt.figure(figsize=(10, 6))
#     plt.suptitle('Performance Metrics', fontsize=16)
#     plot_metric_subplot(train_losses, val_losses, 'Loss', 221)
#     plot_metric_subplot(train_accuracies, val_accuracies, 'Accuracy', 222)
#     plot_metric_subplot(train_aurocs, val_aurocs, 'AUROC', 223)
#     plot_metric_subplot(train_auprcs, val_auprcs, 'AUPRC', 224)
#     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
#     plt.show()
