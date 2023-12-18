# Readme.md

# ****심전도 (Electrocardiogram) 데이터를 활용한 심방세동(Atrial fibrillation) 검측 모델 개발****

### Team : `하버드(HeartBud)`

### Member : ANNIE, ASHTON, JWON

### Duration : 2023/11/6 - 2023/12/15

### 대용량 파일 업데이트 안되어 있음
- PTB_XL과  SPH데이터가 빠져 있음.
  아래에서 다운로드하여 각각의 data폴더에 넣어줘야함.
- 각 py혹은 ipynb파일에서 customfile의 주석을 풀고 생성해주어야함

----

1. PTB-XL, a large publicly available electrocardiography dataset

    ◦ URL : https://physionet.org/content/ptb-xl/1.0.3/

    ◦ 다운로드 : 홈페이지 하단의 ‘Download the ZIP file’ 버튼을 누르거나,

     wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ 명령어를 이용하여 다운로드.

2. SPH, A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements

    ◦ URL : https://doi.org/10.6084/m9.figshare.c.5779802.v1

    ◦ 다운로드 : 홈페이지 하단의 ‘ECG records’ 와 ‘The attributes of ECG records’ 파일을 직접 다운로드.


### DL모델
![스크린샷 2023-12-18 오후 3.35.58.png](README/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-12-18_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_3.35.58.png)
