import argparse
from annie_model_train import train_model
from annie_model_test import test_model_ptb, test_model_sph



def main(train, test):
    if train:
        print("Starting model training...")
        train_model()  # 모델 학습
        print("Training completed.")

    if test:
        print("Starting model testing...")
        test_model_ptb()   # PTB 테스트 데이터셋으로 모델 평가
        test_model_sph()   # SPH 데이터셋으로 모델 평가
        print("Testing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and/or test the CNN model.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()

    try:
        main(args.train, args.test)
    except Exception as e:
        print(f"Error occurred: {e}")
