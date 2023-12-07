import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import ast


def load_ptb_data():
    # PTB-XL 데이터셋 로드 및 전처리
    df_ptb = pd.read_csv('./ptb_xl_data/ptbxl_database.csv', index_col='ecg_id')
    df_ptb.scp_codes = df_ptb.scp_codes.apply(lambda x: ast.literal_eval(x))
    df_ptb.scp_codes = df_ptb.scp_codes.apply(lambda x: list(x.keys()))
    df_ptb['label'] = df_ptb.scp_codes.apply(lambda arr: 1 if 'AFIB' in arr else 0)
    # 총 21799개 중 AFIB 라벨이 있는 샘플은 1514개로 6.95%에 해당, 샘플 불균형 존재

    labels = df_ptb['label'].values
    lead1_signals = np.load('./custom_file/annie_ptb_xl_lead1.npy')


    # 데이터 정규화 (전체)
    lead1_signals_normalized = (lead1_signals - lead1_signals.mean()) / (lead1_signals.std()+1e-7)

    # 데이터셋을 텐서로 변환
    X = torch.Tensor(lead1_signals_normalized)
    y = torch.Tensor(labels).long()

    # 데이터 분할 (훈련:검증:테스트 = 8:1:1)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 채널 수를 1로 추가
    X_train = X_train.unsqueeze(1)
    X_val = X_val.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    return X_train, X_val, X_test, y_train, y_val, y_test



def create_dataloaders(batch_size=64):
    # DataLoader 생성, batch_size를 변경하려면 함수 호출 시 파라미터 변경
    X_train, X_val, X_test, y_train, y_val, y_test = load_ptb_data()

    # DataLoader 생성
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader



def load_sph_data(batch_size=64):
    # SPH 데이터 로드 및 전처리
    df_sph = pd.read_csv("./sph_data/metadata.csv", index_col='ECG_ID')
    # 'AHA_Code' 컬럼의 각 값에 대해 '50'이 포함되어 있는지 확인하고, 'label' 컬럼 생성
    def check_contains_50(code):
        numbers = code.replace(' ', '').replace('+', ';').split(';')
        return '50' in numbers

    df_sph['label'] = df_sph['AHA_Code'].apply(check_contains_50).astype(int)
    
    sph_labels = df_sph['label'].values

    sph_signals = np.load('./custom_file/annie_sph_lead1.npy')
    sph_signals_float = sph_signals.astype(np.float32)
    mean = np.mean(sph_signals_float)
    std = np.std(sph_signals_float)
    sph_signals_normalized = (sph_signals_float - mean) / (std + 1e-7)


    # 데이터 정규화 및 텐서 변환
    X_sph = torch.Tensor(sph_signals_normalized)
    y_sph = torch.Tensor(df_sph['label'].values).long()

    X_sph = X_sph.unsqueeze(1)

    sph_data = TensorDataset(X_sph, y_sph)
    sph_loader = DataLoader(sph_data, batch_size=batch_size)

    return sph_loader

