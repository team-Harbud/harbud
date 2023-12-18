# Readme.md

# ****심전도 (Electrocardiogram) 데이터를 활용한 심방세동(Atrial fibrillation) 검측 모델 개발****

### Team : `하버드(HeartBud)`

### Member : 장보경(ANNIE),  심성식(ASHTON),  정충원(JWON)

### Duration : 2023/11/6 - 2023/12/15

### 대용량 파일 업데이트 안되어 있음
- PTB_XL과  SPH데이터가 빠져 있음.
  아래에서 다운로드하여 각각의 data폴더에 넣어줘야함.
- 각 py혹은 ipynb파일에서 customfile의 주석을 풀고 생성해주어야함

---

1. PTB-XL, a large publicly available electrocardiography dataset

    ◦ URL : https://physionet.org/content/ptb-xl/1.0.3/

    ◦ 다운로드 : 홈페이지 하단의 ‘Download the ZIP file’ 버튼을 누르거나,

     wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ 명령어를 이용하여 다운로드.

2. SPH, A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements

    ◦ URL : https://doi.org/10.6084/m9.figshare.c.5779802.v1

    ◦ 다운로드 : 홈페이지 하단의 ‘ECG records’ 와 ‘The attributes of ECG records’ 파일을 직접 다운로드.
---

## 문제 배경 및 정의 (DS)

### 문제 정의

심혈관 질환(CVDs)은  사망률이 가장 높은 질병 중 하나이며[1], 최근 고소득 국가에서는 암을 추월한 바가 있다.[2] 

심방 세동은 심방의 수축이 소실되어 불규칙하게 수축하는 상태로, 이는 부정맥의 일종인데, 심방세동을 진단하기 위한 가장 정확한 검사 방법은 심전도이다. 심방세동이 있을경우 심전도에서 불규칙한 맥박 및 불규칙한 이소성 P파와 QRS군이 관찰된다.

심전도(ECG)같은 경우 환자의 전반적인 심장 상태를 평가할 수 있는 매우 일반적인 비침습적 도구로써 심혈관 질환(CVDs)진단을 위한 1차 검사로 사용되는 것에도 불구하고 일반의 또는 긴급하게 심전도를 해석해야 하는 응급실 의사에게는 더더욱 어려운 작업이다. [3][4]

 특히나 앞으로 더욱 커질 원격의료 분야에서의 심방세동 탐지를 위한 심전도 모니터링은 그 중요성이 한층 더 높아질 것으로 예상된다. 

---

### 문제 해결의 필요성

 심방세동은 그 위험성 때문에 조기 진단이 중요하다. 하지만 중요도가 높아진 만큼 수요가 계속 증가하는 상황에서 현재의 한정적인 의료 자원으로 심전도 검사 결과를 신속하게 해석하는 것에 어려움을 겪고 있다. 

 이러한 현실 속에서 딥러닝 기반으로 이 문제를 접근하는 방식은 전문 의료인의 의사결정을 도울 뿐만 아니라 해당 분야에서 의료자원을 효율적으로 활용 할 수 있게되어 전체적인 의료질의 향상과 비용절감을 가져올 수 있고 무엇보다 자동 심전도 해석 알고리즘에 기반한 고급 의사 결정 지원 시스템의 지원을 받은 의료진은 심방세동 위험군에 속한 환자를 조기에 발견하여 치료에 기여 할 수 있다.

---

## 데이터(DS)

### 활용 데이터 및 데이터 활용 계획

- **PTB-XL** 의 데이터를 이용하여 **Train, Valid, Test (Internal validation)**을 진행한다.
    
    10초 길이의 18,885명의 환자로부터 얻은 21,837개의 기록으로 구성된 현재 최대 규모의 자유롭게 접근 가능한 임상 12리드 ECG 파형 데이터 세트
    
- **SPH** 의 데이터는 **Test (External validation)** 으로 활용한다.

1. PTB-XL, a large publicly available electrocardiography dataset ([https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/))
2. SPH, A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements ([https://doi.org/10.6084/m9.figshare.c.5779802.v1](https://doi.org/10.6084/m9.figshare.c.5779802.v1))

![Untitled](README/Untitled.png)

- **PTB-XL과 SPH의** 심전도 데이터는 12개의 리드가 있다.

       이12개의 리드는 몇개의 전극을 사용해서 측정하는지, 그리고 +극과 -극이 몇개가 어디에 위치 했는지에 따라 나뉘어집니다.

       우리는 이번 프로젝트에서 오른팔에 (-) → 왼팔(+)의 2개 전극으로 측정한 l 리드데이터를 사용하며, 

        AFIB( 'Atrial Fibrillation', 심방 세동)여부를 1,0으로 라벨링하여 예측한다.

![Untitled](README/Untitled%201.png)

![Untitled](README/Untitled%202.png)


## 평  가

Confusion matrix, Sensitivity(Recall), Specificity, **AUROC (Area Under the Receiver Operating Characteristics)** 등 이진 분류 성능 평가에 사용되는 지표 활용

![Untitled](README/Untitled%203.png)


## 활용 방안 및 기대효과

 전문 의료인의 의사결정을 도울 뿐만 아니라 해당 분야에서 의료 자원을 효율적으로 활용할 수 있게 하여 전체적인 의료질 향상과 비용절감을 가져오고

웨어러블 기기나, 가정용 의료기기 혹은 원격 진료 분야에서 심방세동을 포함한 심혈관 질환(CVDs)을 가진 사람을 조기 발견하여 적기에 치료가 이루어질 수 있도록 도울 수 있다.



----
## DL모델 결과
![스크린샷 2023-12-18 오후 3.35.58.png](README/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-12-18_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_3.35.58.png)
