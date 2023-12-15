import optuna
from a_train_eval import objective
from a_test import test_datasets
from a_models import Custom1DCNN  
from a_visualization import plot_metric_subplot, load_metrics
import matplotlib.pyplot as plt


def train_hyperparameters():
    """
    하이퍼파라미터 튜닝 함수: 하이퍼파라미터 튜닝을 수행합니다.
    """
    print("시작: 하이퍼파라미터 튜닝...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=2)

    # 최적의 하이퍼파라미터 출력
    best_trial = study.best_trial
    print(f"최고 성능을 보인 시도: {best_trial.number+1}")
    print("최적의 하이퍼파라미터:", best_trial.params)

    return best_trial

def main(train=True, test=True):
    """
    메인 함수: 모델 학습, 하이퍼파라미터 튜닝 및 테스트를 수행합니다.
    """
    best_trial = None

    if train:
        # 하이퍼파라미터 튜닝
        best_trial = train_hyperparameters()

    if test and best_trial is not None:
        # 모델 테스트
        print("\n시작: 모델 테스트...")
        model_path = 'best_model.pth'
        test_datasets(Custom1DCNN, model_path)

        # 성능 지표 그래프 및 표시
        train_losses, val_losses, train_accuracies, val_accuracies, train_aurocs, val_aurocs, train_auprcs, val_auprcs = load_metrics(f'trial_{best_trial.number+1}_performance.json')

        plt.figure(figsize=(10, 6))
        plt.suptitle('Performance Metrics', fontsize=16)
        plot_metric_subplot(train_losses, val_losses, 'Loss', 221)
        plot_metric_subplot(train_accuracies, val_accuracies, 'Accuracy', 222)
        plot_metric_subplot(train_aurocs, val_aurocs, 'AUROC', 223)
        plot_metric_subplot(train_auprcs, val_auprcs, 'AUPRC', 224)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
        plt.savefig('1.png')
        plt.show()

if __name__ == "__main__":
    main()
