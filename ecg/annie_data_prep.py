import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    # 데이터 로드
    df_ptb = pd.read_csv('./ptb_xl_data/ptbxl_database.csv', index_col='ecg_id')
    df_ptb.scp_codes = df_ptb.scp_codes.apply(lambda x: ast.literal_eval(x))
    df_ptb.scp_codes = df_ptb.scp_codes.apply(lambda x: list(x.keys()))
    df_ptb['label'] = df_ptb.scp_codes.apply(lambda arr: 1 if 'AFIB' in arr else 0)
    # 21799개 중 1514개. 6.95% 샘플 불균형

    labels = df_ptb['label'].values
    lead1_signals = np.load('./custom_file/annie_ptb_xl_lead1.npy')


    # 데이터 정규화
    lead1_signals_normalized = (lead1_signals - lead1_signals.mean()) / (lead1_signals.std()+1e-7)

    # 데이터셋을 텐서로 변환
    X = torch.Tensor(lead1_signals_normalized)
    y = torch.Tensor(labels).long()

    # 데이터 분할 (8:1:1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 채널 수를 1로 추가
    X_train = X_train.unsqueeze(1)
    X_val = X_val.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(batch_size=64):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # DataLoader 생성
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# 질문
# batch_size도 바꾸고 싶을 땐 어떻게 해야하는가