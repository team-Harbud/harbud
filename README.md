# 하버드 최종 결과 보고서

[data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)

# [0] 기본

- Pytorch
    - Pytorch official tutorial
        
        [https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)
        
    - Pytorch tensor
        
        
        ### 텐서의 종류
        
        [https://velog.io/@olxtar/PyTorch-Tensor의-Type](https://velog.io/@olxtar/PyTorch-Tensor%EC%9D%98-Type)
        
        - long tensor: 64비트 정수로 바꾸기
        
    - Pytorch 사용법
        
        
        책: 실전 파이토치 딥러닝 프로젝트 
        
        공식문서: 
        
        [https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)
        
        강의: 
        
        유데미- 딥러닝의 모든것
        
        [https://kmooc.udemy.com/course/best-artificial-neural-networks/learn/lecture/39178006?start=0#content](https://kmooc.udemy.com/course/best-artificial-neural-networks/learn/lecture/39178006?start=0#content)
        
        패스트 캠퍼스- 처음부터 시작하는 딥러닝 유치원 
        
        [https://fastcampus.app/course-detail/203355](https://fastcampus.app/course-detail/203355)
        
    - Windows에서 Cuda설치
        
        [https://teang1995.tistory.com/17](https://teang1995.tistory.com/17)
        
        ### 1. 내 GPU 확인
        
        어렵지 않다.
        
        ![https://blog.kakaocdn.net/dn/QIxAk/btrpYlZ2A1S/QGbBaxQ6LdkjwGO9h1Mh8K/img.png](https://blog.kakaocdn.net/dn/QIxAk/btrpYlZ2A1S/QGbBaxQ6LdkjwGO9h1Mh8K/img.png)
        
        우선 시작 메뉴 옆의 검색창에서 장치 관리자를 검색한 뒤, 디스플레이 어댑터를 확인한다.
        
        아하, 이 컴퓨터에 달린 그래픽 카드는 GTX 1660이구나.
        
        [https://www.wikiwand.com/en/CUDA](https://www.wikiwand.com/en/CUDA)
        
        위의 주소에서 본인의 그래픽 카드명을 검색해보면
        
        ![https://blog.kakaocdn.net/dn/EfSqP/btrpVwnts9s/5qfBK66Mulf8RWKeZke6b1/img.png](https://blog.kakaocdn.net/dn/EfSqP/btrpVwnts9s/5qfBK66Mulf8RWKeZke6b1/img.png)
        
        순서대로 Compute Capability, Micro-Architecture, GPU, GeForce, Quadro/NVS, Tesla, Tegra.. 라고 한다.
        
        내 그래픽 카드의 Compute Capability가 7.5임을 알 수 있다.
        
        [https://en.wikipedia.org/wiki/CUDA#GPUs_supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
        
        에 접속 후, 본인의 GPU의 Compute Capability에 맞는 CUDA SDK 버전을 확인.
        
        ![https://blog.kakaocdn.net/dn/cECtX9/btrpU0I4p0z/KydPpdMG3PFl8AaO7MOtg1/img.png](https://blog.kakaocdn.net/dn/cECtX9/btrpU0I4p0z/KydPpdMG3PFl8AaO7MOtg1/img.png)
        
        GTX 1660의 Compute Capability는 7.5이니 여기에 맞는 CUDA11.1을 사용하도록 하겠다.
        
        ### 2. CUDA 설치
        
        [https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10)
        
        위 링크로 들어가 본인의 환경에 맞게 옵션을 선택한다.
        
        ![https://blog.kakaocdn.net/dn/obdVf/btrpVvWpLS3/8QT9b7gpixtXLZXxVoYWDK/img.png](https://blog.kakaocdn.net/dn/obdVf/btrpVvWpLS3/8QT9b7gpixtXLZXxVoYWDK/img.png)
        
        윈도우10, 쿠다11.1, exe
        
        그러면 설치 파일을 다운 받을 수 있는 버튼, 혹은 링크가 등장하니 이를 설치! 동의, 예~~
        
        ![https://blog.kakaocdn.net/dn/ckJtwA/btrpU2mHDxs/UIqQsLSXGMgYOM4CG4Wx2K/img.png](https://blog.kakaocdn.net/dn/ckJtwA/btrpU2mHDxs/UIqQsLSXGMgYOM4CG4Wx2K/img.png)
        
        설치 완!
        
        ### 3. cuDNN 설치
        
        [https://developer.nvidia.com/rdp/cudnn-download#a-collapse714-92](https://developer.nvidia.com/rdp/cudnn-download#a-collapse714-92)
        
        위 링크에 접속해 NVIDIA developer 에 가입.
        
        가입 후 설치를 시도하면
        
        ![https://blog.kakaocdn.net/dn/cDOyMB/btrpWzc9GRV/DhfBbnaIQHVUIk6rnKUwe0/img.png](https://blog.kakaocdn.net/dn/cDOyMB/btrpWzc9GRV/DhfBbnaIQHVUIk6rnKUwe0/img.png)
        
        내가 가진 CUDA에 맞춰 추천을 해주는 건지, 최신 버전이라 추천을 해주는 것인지는 모르겠지만 11.5가 바로 추천돼있으니 이를 다운 받는다.
        
        ![https://blog.kakaocdn.net/dn/qmCS8/btrp008HdHQ/nKnruV1dDooPyglNQCw8A0/img.png](https://blog.kakaocdn.net/dn/qmCS8/btrp008HdHQ/nKnruV1dDooPyglNQCw8A0/img.png)
        
        그럼 요렇게 폴더가 생기는 것을 확인할 수 있는데, `bin`, `include`, `lib` 안의 파일들을 각각 앞서 설치한 CUDA의 경로에 알맞은 위치에 넣어주어야 한다.
        
        알맞은 경로는 환경 변수에서 확인할 수 있는데,
        
        ![https://blog.kakaocdn.net/dn/bd9m0U/btrpWzqGGaR/8jN4WehX2U58wvZvykTyZk/img.png](https://blog.kakaocdn.net/dn/bd9m0U/btrpWzqGGaR/8jN4WehX2U58wvZvykTyZk/img.png)
        
        CUDA 환경변수
        
        CUDA_PATH, CUDA_PATH_V11_5 의 변수 값에 지정된 경로의 `bin`, `include`, `lib` 에 cudnn의 파일들을 맞게 넣어주면 된다.
        
        ### 4. Torch 설치
        
        ![https://blog.kakaocdn.net/dn/PNdjh/btrp0If8G8Y/8bhsgfJl28ts9lIKc09irK/img.png](https://blog.kakaocdn.net/dn/PNdjh/btrp0If8G8Y/8bhsgfJl28ts9lIKc09irK/img.png)
        
        Pytorch 공홈 설치 옵션
        
        알맞게 옵션을 지정해준 뒤, 명령어를 그대로 실행.
        
        **했는데, torch 에서 cuda를 못 잡는 경우가 있다.**
        
        **이런 경우, 환경 변수에서**
        
        `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\libnvvp`
        
        `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin`
        
        **을 추가해주고, 리부트하면 잘 잡힌다.**
        
        ![https://blog.kakaocdn.net/dn/bb7HA3/btrp2B9aqvX/fP4UDE5xB1XpkXLbUXcpkk/img.png](https://blog.kakaocdn.net/dn/bb7HA3/btrp2B9aqvX/fP4UDE5xB1XpkXLbUXcpkk/img.png)
        
        짝짝짝
        
        ### References
        
        [https://mickael-k.tistory.com/17](https://mickael-k.tistory.com/17)
        
        [https://mickael-k.tistory.com/18](https://mickael-k.tistory.com/18)
        
    - Mac에서의 Cuda사용법(mps)
        
        [https://discuss.pytorch.kr/t/apple-m1-pytorch-gpu/276](https://discuss.pytorch.kr/t/apple-m1-pytorch-gpu/276)
        
        Apple M1 칩에서의 GPU 가속 기능이 드디어,
        
        **PyTorch 1.12부터 가능**
        
        해진다고 합니다!
        
        ![https://discuss.pytorch.kr/images/emoji/apple/tada.png?v=12](https://discuss.pytorch.kr/images/emoji/apple/tada.png?v=12)
        
        기존의 `cuda` 장치처럼 `mps`(Apple의 Metal Performance Shaders) 장치로 접근해서 사용할 수 있다고 합니다.
        
        (아래 코드는 [MPS Backend 문서 341](https://pytorch.org/docs/master/notes/mps.html?utm_source=pytorchkr)에서 가져왔습니다.)
        
        ```python
        
        mps_device = torch.device("mps")
        
        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        x = torch.ones(5, device="mps")
        
        # Any operation happens on the GPU
        y = x * 2
        
        # Move your model to mps just like any other device
        model = YourFavoriteNet()
        model.to(mps_device)
        
        # Now every call runs on the GPU
        pred = model(x)
        
        ```
        
        아래와 같이 M1 Ultra에서 학습과 평가 모두 CPU 대비 월등한 성능 향상을 보여주는 것을 보실 수 있습니다.
        
        ![https://discuss.pytorch.kr/images/emoji/apple/slight_smile.png?v=12](https://discuss.pytorch.kr/images/emoji/apple/slight_smile.png?v=12)
        
        **[image1920×1078 195 KB](https://discuss.pytorch.kr/uploads/default/original/1X/b36a3bb996c0dc654e6d39b08a8a57c7761035a3.jpeg)**
        
        ![https://discuss.pytorch.kr/uploads/default/optimized/1X/b36a3bb996c0dc654e6d39b08a8a57c7761035a3_2_690x387.jpeg](https://discuss.pytorch.kr/uploads/default/optimized/1X/b36a3bb996c0dc654e6d39b08a8a57c7761035a3_2_690x387.jpeg)
        
        자세한 공지는 아래에서 보실 수 있으며, [Preview 버전(Nightly Build) 143](https://pytorch.kr/get-started/locally/)에서는 바로 사용할 수 있다고 합니다.
        
        [pytorch.org](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/?utm_source=pytorchkr)
        **[Introducing Accelerated PyTorch Training on Mac 353](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/?utm_source=pytorchkr)**
        In collaboration with the Metal engineering team at Apple, we are excited to announce support for GPU-accelerated PyTorch training on Mac. Until now, PyTorch training on Mac only leveraged the CPU, but with the upcoming PyTorch v1.12 release,...
        
        ![https://discuss.pytorch.kr/uploads/default/original/1X/75bff5469a36bcf10d57f28a3f077a7dd869e43a.png](https://discuss.pytorch.kr/uploads/default/original/1X/75bff5469a36bcf10d57f28a3f077a7dd869e43a.png)
        
        [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/?utm_source=pytorchkr](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/?utm_source=pytorchkr)
        
        기존의 많은 사용자들의 요청과 아우성(…)은 아래 이슈에서 보실 수 있습니다
        
        ![https://discuss.pytorch.kr/images/emoji/apple/roll_eyes.png?v=12](https://discuss.pytorch.kr/images/emoji/apple/roll_eyes.png?v=12)
        
        [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/issues/47702?utm_source=pytorchkr)
        **[GPU acceleration for Apple's M1 chip? 102](https://github.com/pytorch/pytorch/issues/47702?utm_source=pytorchkr)**opened  2020년 11월 11일 [dexios1](https://github.com/dexios1?utm_source=pytorchkr)module: performance triaged
        ## 🚀 Feature
        Hi, 
        
        I was wondering if we could evaluate PyTorch's performance[…](https://discuss.pytorch.kr/t/apple-m1-pytorch-gpu/276)
        
        ![https://discuss.pytorch.kr/uploads/default/original/1X/4ca895dcfe56b27033962dfd28a3db90cecd4221.jpeg](https://discuss.pytorch.kr/uploads/default/original/1X/4ca895dcfe56b27033962dfd28a3db90cecd4221.jpeg)
        
    - Cuda와 Pytorch 버전 맞추기 과정에서 서버 폴더 용량 초과 문제 해결
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%201.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%202.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%203.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%204.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%205.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%206.png)
        
        PyTorch를 `/dev/xvdb1` 파티션에 설치하는 경우, 필수적으로 실험 데이터 파일을 모두 옮길 필요는 없습니다. 하지만, PyTorch 및 관련 라이브러리가 설치된 파티션과 데이터 파일이 저장된 파티션 간의 데이터 액세스 효율성을 고려해야 합니다. 다음과 같이 진행할 수 있습니다:
        
        1. **Python 가상 환경 생성**:
            - `/dev/xvdb1`에 Python 가상 환경을 생성합니다. 이를 위해 먼저 해당 파티션으로 이동한 후, 가상 환경을 생성합니다:
                
                ```bash
                cd /dev/xvdb1
                python -m venv myenv
                
                ```
                
        2. **가상 환경 활성화 및 PyTorch 설치**:
            - 생성된 가상 환경을 활성화하고, PyTorch를 설치합니다:
                
                ```bash
                source myenv/bin/activate
                pip install torch torchvision torchaudio
                
                ```
                
        3. **데이터 파일 접근**:
            - PyTorch와 데이터 파일이 다른 파티션에 있더라도, Python 스크립트나 Jupyter 노트북 등에서 해당 파일들을 정상적으로 읽고 쓸 수 있습니다.
            - 데이터 파일의 경로를 올바르게 지정하면, 가상 환경에서 실행되는 PyTorch 프로그램이 이 파일들을 접근할 수 있습니다.
        4. **성능 고려**:
            - 만약 데이터 파일이 매우 크거나, 데이터 I/O가 빈번하게 발생하는 경우, 데이터 파일을 PyTorch가 설치된 같은 파티션(`/dev/xvdb1`)으로 옮기는 것이 성능상 이점이 있을 수 있습니다.
            - 하지만 일반적인 경우에는 데이터 파일을 별도의 파티션에 두어도 문제가 되지 않습니다.
        
        즉, PyTorch를 다른 파티션에 설치하는 것 자체가 데이터 파일을 이동해야 하는 필요성을 직접적으로 야기하지는 않지만, 데이터 액세스의 효율성과 성능을 고려하여 결정하는 것이 좋습니다.
        
        ---
        
        ## 가상환경 삭제
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%207.png)
        
    
- 가상환경 및 Server
    - Server사용법
        
        하버드 팀에게 지원되는 서버는 T4 1개 서버 이며, 접속 정보는 다음과 같습니다.
        
        - IP : 223.130.139.142
        - user : root
        - password : E5g2LHLUF+h
        - port : 22 (ssh 기본 포트번호)
        - host : **annie**
        
        하나의 자원을 팀원끼리 나눠서 사용해야하니, **root 권한으로 접속하여 사용자 계정을 만들어서 각자 접속**하여 사용하시는 것을 추천드립니다.(번거로울 수 있으나, 한정된 GPU 사용을 팀원분들께서 조정하여 사용하시기 바랍니다.)
        
        ---
        
        사용자 계정은 아직!
        
        **사용자 계정 만드는법**
        
        1.루트 권한으로 로그인
        
        su -
        
        2.새로운 사용자 계정 만들기
        
        adduser <your-username>
        
        3.Sudo 권한 부여 (선택 사항)
        
        usermod -aG sudo <your-username>
        
        4.로그아웃 및 새로운 사용자로 로그인
        
        exit
        
        su - <your-username>
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%208.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%209.png)
        
    - Server 폴더 변경 후 가상환경 재설치
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2010.png)
        
        Collecting torch
        Using cached torch-2.1.1-cp38-cp38-manylinux1_x86_64.whl (670.2 MB)
        Collecting nvidia-cusparse-cu12==12.1.0.106; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
        Collecting typing-extensions
        Downloading typing_extensions-4.9.0-py3-none-any.whl (32 kB)
        Collecting sympy
        Using cached sympy-1.12-py3-none-any.whl (5.7 MB)
        Collecting nvidia-cuda-nvrtc-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
        Collecting nvidia-cublas-cu12==12.1.3.1; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
        Collecting fsspec
        Using cached fsspec-2023.12.1-py3-none-any.whl (168 kB)
        Collecting nvidia-cufft-cu12==11.0.2.54; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
        Collecting nvidia-cuda-cupti-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
        Collecting nvidia-curand-cu12==10.3.2.106; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
        Collecting triton==2.1.0; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached triton-2.1.0-0-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.2 MB)
        Collecting nvidia-cudnn-cu12==8.9.2.26; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
        Collecting networkx
        Using cached networkx-3.1-py3-none-any.whl (2.1 MB)
        Collecting jinja2
        Using cached Jinja2-3.1.2-py3-none-any.whl (133 kB)
        Collecting nvidia-nvtx-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
        Collecting filelock
        Using cached filelock-3.13.1-py3-none-any.whl (11 kB)
        Collecting nvidia-cusolver-cu12==11.4.5.107; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
        Collecting nvidia-cuda-runtime-cu12==12.1.105; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
        Collecting nvidia-nccl-cu12==2.18.1; platform_system == "Linux" and platform_machine == "x86_64"
        Using cached nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
        ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
        
        ---
        
        ## 엔비디아 드라이버 링크를 서버에서 재설치
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2011.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2012.png)
        
    - 가상환경 설정 후 라이브러리 재설치
        
        ## 가상환경 설정
        
        [https://theorydb.github.io/dev/2020/02/14/dev-dl-setting-local-python/](https://theorydb.github.io/dev/2020/02/14/dev-dl-setting-local-python/)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2013.png)
        
        ## torch 재설치
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2014.png)
        
    - Root 사용자 → 가상환경
        
        <aside>
        💡 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: [https://pip.pypa.io/warnings/venv](https://pip.pypa.io/warnings/venv)
        
        </aside>
        
        이 경고 메시지는 `pip`를 'root' 사용자로 실행할 때 발생할 수 있는 문제에 대해 경고하고 있습니다. 일반적으로 Linux 시스템에서 'root' 사용자로 라이브러리를 설치하면, 시스템 권한과 충돌하거나 권한 문제를 일으킬 수 있습니다. 또한, 시스템 패키지 관리자와 충돌할 수 있습니다. 이러한 문제를 피하기 위해, Python의 가상 환경을 사용하는 것이 권장됩니다.
        
        Python 가상 환경을 사용하면 다음과 같은 이점이 있습니다:
        
        1. **시스템 격리**: 가상 환경은 프로젝트별로 독립된 Python 환경을 제공하여, 시스템의 나머지 부분과 격리됩니다. 이는 패키지 충돌을 방지하고, 프로젝트별로 다른 패키지 버전을 유지할 수 있게 해줍니다.
        2. **권한 문제 방지**: 'root' 사용자로 패키지를 설치하지 않아도 되므로, 권한 문제를 예방할 수 있습니다.
        
        가상 환경을 만들고 사용하는 방법은 다음과 같습니다:
        
        1. **가상 환경 생성**:
            
            ```bash
            python3 -m venv myenv
            
            ```
            
        2. **가상 환경 활성화**:
            - Linux나 macOS에서:
                
                ```bash
                source myenv/bin/activate
                
                ```
                
            - Windows에서:
                
                ```bash
                myenv\\Scripts\\activate
                
                ```
                
        3. **가상 환경 내에서 패키지 설치**:
            
            ```bash
            pip install 필요한_패키지
            
            ```
            
        4. **가상 환경 비활성화**:
            
            ```bash
            deactivate
            
            ```
            
        
        이렇게 하면 각 프로젝트에 필요한 독립적인 환경을 만들 수 있으며, 시스템 레벨의 Python 환경에 영향을 주지 않고 패키지를 관리할 수 있습니다.
        
        ---
        
        실행 후 재오류
        
        <aside>
        💡 root@ds-server009:~/harbud/ecg# python3 -m venv myenv
        The virtual environment was not created successfully because ensurepip is not
        available.  On Debian/Ubuntu systems, you need to install the python3-venv
        package using the following command.
        
        ```
        apt install python3.8-venv
        
        ```
        
        You may need to use sudo with that command.  After installing the python3-venv
        package, recreate your virtual environment.
        
        Failing command: ['/root/harbud/ecg/myenv/bin/python3', '-Im', 'ensurepip', '--upgrade', '--default-pip']
        
        </aside>
        
        이 메시지는 Python의 가상 환경을 생성하려고 했으나 필요한 `ensurepip` 패키지가 없어서 가상 환경이 성공적으로 생성되지 않았다는 것을 알려주고 있습니다. Debian/Ubuntu 시스템에서는 `python3-venv` 패키지를 설치해야 Python 가상 환경을 생성할 수 있습니다.
        
        해결 방법은 다음과 같습니다:
        
        1. **python3-venv 패키지 설치**:
            - 해당 시스템이 Ubuntu 또는 Debian 기반인 경우, `python3-venv` 패키지를 설치해야 합니다. 명령어는 다음과 같습니다:
                
                ```bash
                sudo apt install python3.8-venv
                
                ```
                
            - 위 명령어에서 `python3.8-venv`는 Python 3.8 버전을 기준으로 하고 있습니다. 사용 중인 Python 버전에 맞게 명령어를 조정해야 할 수도 있습니다.
        2. **가상 환경 다시 생성**:
            - `python3-venv` 패키지 설치 후, 가상 환경을 다시 생성합니다:
                
                ```bash
                python3 -m venv myenv
                
                ```
                
        3. **가상 환경 활성화 및 사용**:
            - 생성된 가상 환경을 활성화하고, 필요한 Python 패키지들을 설치하여 사용합니다.
        
        가상 환경을 사용하면 시스템 전역 환경을 변경하지 않고도 필요한 Python 패키지들을 설치하고 관리할 수 있어, 보다 안정적인 개발 환경을 구축할 수 있습니다.
        
    - Server 자원상황
        
        서버 관련 명령어 입니다.
        
        CPU 정보 확인:
        lscpu
        
        메모리(RAM) 정보 확인:
        free -h
        
        실시간 CPU 사용 현황 확인:
        top
        
        실시간 메모리 사용 현황 확인:
        htop
        
        GPU 상태 및 사용 중인 프로세스 확인:
        nvidia-smi
        
        ```bash
        root@ds-server009:~/harbud# nvidia-smi
        Sat Nov 25 02:16:26 2023       
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 470.199.02   Driver Version: 470.199.02   CUDA Version: 11.4     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |                               |                      |               MIG M. |
        |===============================+======================+======================|
        |   0  Tesla T4            On   | 00000000:00:05.0 Off |                  Off |
        | N/A   63C    P0    60W /  70W |  12446MiB / 16127MiB |     97%      Default |
        |                               |                      |                  N/A |
        +-------------------------------+----------------------+----------------------+
                                                                                       
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A    687505      C   /bin/python3                      211MiB |
        |    0   N/A  N/A    707034      C   /bin/python3                     6087MiB |
        |    0   N/A  N/A    707100      C   /bin/python3                     5713MiB |
        |    0   N/A  N/A    707191      C   /bin/python3                      215MiB |
        |    0   N/A  N/A    707214      C   /bin/python3                      215MiB |
        +-----------------------------------------------------------------------------+
        root@ds-server009:~/harbud# lscpu
        Architecture:                    x86_64
        CPU op-mode(s):                  32-bit, 64-bit
        Byte Order:                      Little Endian
        Address sizes:                   46 bits physical, 48 bits virtual
        CPU(s):                          8
        On-line CPU(s) list:             0-7
        Thread(s) per core:              1
        Core(s) per socket:              8
        Socket(s):                       1
        NUMA node(s):                    1
        Vendor ID:                       GenuineIntel
        CPU family:                      6
        Model:                           85
        Model name:                      Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
        Stepping:                        7
        CPU MHz:                         2194.929
        BogoMIPS:                        4389.83
        Hypervisor vendor:               Xen
        Virtualization type:             full
        L1d cache:                       256 KiB
        L1i cache:                       256 KiB
        L2 cache:                        8 MiB
        L3 cache:                        198 MiB
        NUMA node0 CPU(s):               0-7
        Vulnerability Itlb multihit:     KVM: Vulnerable
        Vulnerability L1tf:              Mitigation; PTE Inversion
        Vulnerability Mds:               Mitigation; Clear CPU buffers; SMT Host state unknown
        Vulnerability Meltdown:          Mitigation; PTI
        Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
        Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
        top - 02:17:14 up 8 days,  9:49,  0 users,  load average: 8.20, 8.17, 7.84
        Tasks: 203 total,  10 running, 193 sleeping,   0 stopped,   0 zombie
        %Cpu(s): 94.7 us,  5.2 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
        MiB Mem :  40106.7 total,  12130.8 free,  26695.2 used,   1280.8 buff/cache
        MiB Swap:      0.0 total,      0.0 free,      0.0 used.  12158.5 avail Mem 
        
            PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                                                           
         707252 root      20   0 9303564   2.6g 892288 R  99.7   6.6   0:53.39 python3                                                                           
         707100 root      20   0   13.8g   2.3g 892740 R  98.7   5.8   2:11.95 python3                                                                           
         707034 root      20   0   10.0g   2.0g 211732 R  98.3   5.2   3:13.06 python3                                                                           
         707235 root      20   0 9303528   2.6g 892100 R  98.3   6.6   1:09.50 python3                                                                           
         707354 root      20   0 8445244   2.1g 743236 R  98.3   5.4   0:21.25 python3                                                                           
         707409 root      20   0 6977520   1.7g 743232 R  98.0   4.4   0:10.35 python3                                                                           
         707248 root      20   0 9303556   2.6g 892604 R  97.7   6.6   0:59.24 python3                                                                           
        root@ds-server009:~/harbud# htop
        root@ds-server009:~/harbud# df -h
        Filesystem      Size  Used Avail Use% Mounted on
        udev             20G     0   20G   0% /dev
        tmpfs           4.0G  836K  4.0G   1% /run
        /dev/xvda1       49G   29G   19G  61% /
        tmpfs            20G  666M   19G   4% /dev/shm
        tmpfs           5.0M     0  5.0M   0% /run/lock
        tmpfs            20G     0   20G   0% /sys/fs/cgroup
        /dev/xvdb1      246G   61M  233G   1% /DATA
        tmpfs           4.0G     0  4.0G   0% /run/user/0
        
        root@ds-server009:~/harbud# free -h
                      total        used        free      shared  buff/cache   available
        Mem:           39Gi        24Gi        13Gi       730Mi       1.2Gi        13Gi
        Swap:            0B          0B          0B
        ```
        
    
- Git
    - Github와 Linux commend
        
        
        - 최초 등록
            
            ```bash
            =========================
            [Git관련 최로 명령어]
            =========================
            git config --list
            git config --global user.name "본인 아이디"
            git config --global user.email "이메일@gmail.com" 
            git config --global core.editor "vim"git log
            git config --global core.pager "cat"
            git config --list
             
            **=========================
            [토큰]
            =========================**
            
            -------------------------
            [Github site에서]
            -------------------------
            -프로필 setting
            -Developer settings
            -Personal access tokens
            -Generate new token
            
            ------------------------
            [터미널에서]
            ------------------------
            처음으로 git push origin main 할때, 토큰을 입력하라고 나옴,
            
            =========================
            branch
            =========================
            브런치 새로만들기
            git branch 새로운_브랜치_이름
            
            만든 브런치로 전환
            git checkout 새로운_브랜치_이름
            혹은, git switch 새로운_브랜치_이름
            ```
            
        - 정리 (깃 연결된 상태에서의 커멘드)
            
            ```bash
            #터미널
            cd harbud     # 깃에서 관리 되고 있는 폴더에 들어가셔야함
            git status    # 현재 깃 상태 알수 있는 커멘드 중간중간 치면서 확인해보시면 됩니다.
            git add .
            git commit
            
            #vim 창이 뜸.
            - i라고 누루면 글 수정가능.
            - 실수로 다른거 눌러서 이상해졌다면 esc 연타~
            - 수정 상태에서 commit메세지.. 무엇을 수정했나~ 세이브파일의 설명이라고 보시면됨
            - 수정후 esc눌러 수정 모드에서 나온후 :wq 라고침(저장하고 종료)
            
            #다시 터미널로 나와서
            git push origin (브런치 이름) 
            # git push origin main # 혼자 쓸때는 아마 이것을 주로 사용했었을 것
            ```
            
        - 블로그와 노션 링크
            - 깃허브
                
                [https://www.notion.so/Git-Hub-bd8d66014e1c432293de368bd6d7fced?pvs=4](https://www.notion.so/Git-Hub-bd8d66014e1c432293de368bd6d7fced?pvs=21)
                
            - 리눅스 기본 명령어
                
                [https://ashton0410.tistory.com/entry/OSLinux-command-정리](https://ashton0410.tistory.com/entry/OSLinux-command-%EC%A0%95%EB%A6%AC)
                
            - vim 기본 명령어
                
                [https://ashton0410.tistory.com/entry/OSLinux-Vim-command-리눅스-빔-명령어](https://ashton0410.tistory.com/entry/OSLinux-Vim-command-%EB%A6%AC%EB%88%85%EC%8A%A4-%EB%B9%94-%EB%AA%85%EB%A0%B9%EC%96%B4)
                
        
        - 깃허브 강의자료 (업로드 중 총 4개파일)
            
            [https://drive.google.com/drive/folders/144cgBjtKyMqKZJiCl_CY-UQd9nOgdYKA?usp=drive_link](https://drive.google.com/drive/folders/144cgBjtKyMqKZJiCl_CY-UQd9nOgdYKA?usp=drive_link)
            
        
    - Github commit, push 안되는 문제 해결
        
        [이홍주1208](https://www.notion.so/1208-e481705a766146cea758de5db7da910f?pvs=21)
        
        - 최초 등록
            
            ```bash
            =========================
            [Git관련 최로 명령어]
            =========================
            git config --list
            git config --global user.name "본인 아이디"
            git config --global user.email "이메일@gmail.com" 
            git config --global core.editor "vim"git log
            git config --global core.pager "cat"
            git config --list
            
            --푸시권한없음--
            git config --local user.name "깃헙아이디"
            git config --local user.email "깃헙이메일"
             
            **=========================
            [토큰]
            =========================**
            
            -------------------------
            [Github site에서]
            -------------------------
            -프로필 setting
            -Developer settings
            -Personal access tokens
            -Generate new token
            
            ------------------------
            [터미널에서]
            ------------------------
            처음으로 git push origin main 할때, 토큰을 입력하라고 나옴,
            
            =========================
            branch
            =========================
            브런치 새로만들기
            git branch 새로운_브랜치_이름
            
            만든 브런치로 전환
            git checkout 새로운_브랜치_이름
            혹은, git switch 새로운_브랜치_이름
            ```
            
        - 정리 (깃 연결된 상태에서의 커멘드)
            
            ```bash
            #터미널
            cd harbud     # 깃에서 관리 되고 있는 폴더에 들어가셔야함
            git status    # 현재 깃 상태 알수 있는 커멘드 중간중간 치면서 확인해보시면 됩니다.
            git add .
            git commit
            
            #vim 창이 뜸.
            - i라고 누루면 글 수정가능.
            - 실수로 다른거 눌러서 이상해졌다면 esc 연타~
            - 수정 상태에서 commit메세지.. 무엇을 수정했나~ 세이브파일의 설명이라고 보시면됨
            - 수정후 esc눌러 수정 모드에서 나온후 :wq 라고침(저장하고 종료)
            
            #다시 터미널로 나와서
            git push origin (브런치 이름) 
            # git push origin main # 혼자 쓸때는 아마 이것을 주로 사용했었을 것
            ```
            
        - 블로그와 노션 링크
            - 깃허브
                
                [https://www.notion.so/Git-Hub-bd8d66014e1c432293de368bd6d7fced?pvs=4](https://www.notion.so/Git-Hub-bd8d66014e1c432293de368bd6d7fced?pvs=21)
                
            - 리눅스 기본 명령어
                
                [https://ashton0410.tistory.com/entry/OSLinux-command-정리](https://ashton0410.tistory.com/entry/OSLinux-command-%EC%A0%95%EB%A6%AC)
                
            - vim 기본 명령어
                
                [https://ashton0410.tistory.com/entry/OSLinux-Vim-command-리눅스-빔-명령어](https://ashton0410.tistory.com/entry/OSLinux-Vim-command-%EB%A6%AC%EB%88%85%EC%8A%A4-%EB%B9%94-%EB%AA%85%EB%A0%B9%EC%96%B4)
                
        
        - 깃허브 강의자료 (업로드 중 총 4개파일)
            
            [https://drive.google.com/drive/folders/144cgBjtKyMqKZJiCl_CY-UQd9nOgdYKA?usp=drive_link](https://drive.google.com/drive/folders/144cgBjtKyMqKZJiCl_CY-UQd9nOgdYKA?usp=drive_link)
            
        
        ---
        
        [https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)
        
        [https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories)
        
        commit 까진 되는데, push 권한이 없어서 계속 막히는 듯 하다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2015.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2016.png)
        
        일단 commit 해놓고, 내일 push 해주시면 확인하기
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2017.png)
        
    - Github 사용법 정리
        
        **ashton 공유**
        
        팀장
        
        Organization을 생성한다.
        팀원 초대를 한다.
        repository를 생성하고, .gitignore 와 대상파일 생성을 완료한다.
        (https://www.toptal.com/developers/gitignore)
        issue template을 만든다.
        완료되었음을 알린다.
        팀원이 생성한 issue를 관리한다.
        팀원-7 이후 PR에 대해 코드리뷰를 실시한다.
        팀원-8(추가작업사항)에 대해 다시 코드리뷰를 실시하여 승인처리한다.
        팀원들에게 업데이트 되었음을 알린다.
        
        팀원
        
        주제에 관한 토론을 실시한다.
        팀원별 기능 분배를 실시한다.
        팀 repo에서 issue를 작성한다.
        팀장-5 후 fork, clone 한다.
        브랜치 생성 후, 작업을 실시한다.
        작업 완료 후 팀 repo의 업데이트사항을 확인 한 후, 내 repo의 브랜치 명으로 push 한다.
        pull request를 open한다.
        코드리뷰 결과 추가 작업사항에 대해 작업 후 작업중인 브랜치로 push하여 해당 사항을 업데이트 한다.
        다른 팀원의 업데이트 사항을 내 작업 브랜치 혹은 main 브랜치로 업데이트(pull or fetch&merge) 하여 최신사항을 업데이트 한다.
        의 과정을 프로젝트가 완료될때 까지 반복한다.
        
        ---
        
        [ashton0410.tistory.com/entry/OSLinux-command-정리](http://ashton0410.tistory.com/entry/OSLinux-command-%EC%A0%95%EB%A6%AC)
        
    - 이홍주 Server, Github 멘토링
    - Github: your branch is ahead of 'upstream/main' by 3 commits.
        - .gitignore에 폴더 지정하는 방법에 문제가 있는것을 확인. →ecg/custom_file/ 현태여야함
        - git log 로 깃 상태 확인
        - (HEAD -> main)은 현재 작업 중인 브랜치
            
            (upstream/main, upstream/HEAD)는 원격 저장소(upstream)의 main 브랜치
            
            어떻게 할지 정해야함.
            최근 브런치들을 지운다면
            현재 파일 따로 저장
            
            ```bash
            git reset --hard HEAD~3
            git push -f origin main
            ```
            
            위 방법이 추천됨
            
            위 내용을 보기전에 아래와 같이 진행함. 
            
            최근 문제되는 commit을 삭제 → 과거 깃 내용을 현재에 덮어씌움
            
            ```bash
            git reset --hard HEAD~3
            git pull upstream main
            ```
            
            ---
            
            문제점
            
            git reset --hard HEAD~3
            
            를 하면서 .gitignore에 지정한 파일까지 없어짐. 파일이 업이안되더라도 존재 하는지는 깃이 아는것같음. 만약 깃에 없는 파일을 다 지운다고 한다면 data파일도 날라가야 하는데 그러지 않았음.
            
            때문에 백업하지 않은 일부 파일 날아감… 복구 가능(보경님 커스텀 파일 ㅠㅠ)
            
    
- 기타
    - json 저장
        
        [https://jsikim1.tistory.com/221](https://jsikim1.tistory.com/221)
        
        ## 첫 성공
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2018.png)
        
    - csv read 속도 올리기
        
        CSV에서 데이터를 빠르게 읽어오기 위해 몇 가지 방법을 고려할 수 있습니다. 아래는 몇 가지 일반적인 전략입니다:
        
        1. **판다스(pandas)의 `read_csv` 최적화 사용:**
            - **`pandas`** 라이브러리는 **`read_csv`** 함수에 여러 가지 최적화 기능을 포함하고 있습니다. **`read_csv`**의 파라미터들을 조정하여 속도를 향상시킬 수 있습니다.
            - **`usecols`**: 필요한 열만 선택하여 읽을 수 있습니다.
            - **`dtype`**: 데이터 타입을 미리 지정하여 메모리 사용량을 줄일 수 있습니다.
            - **`chunksize`**: 데이터를 작은 청크로 나누어 읽을 수 있습니다.
        
        ```python
        pythonCopy code
        import pandas as pd
        
        # 예시
        df = pd.read_csv('your_file.csv', usecols=['col1', 'col2'], dtype={'col1': int, 'col2': float})
        
        ```
        
        1. **Dask 사용:**
            - **`dask`**는 큰 데이터셋을 처리하기 위한 병렬 처리를 지원하는 라이브러리입니다. 판다스와 유사한 API를 가지고 있으며, 큰 데이터를 여러 청크로 나누어 병렬로 처리할 수 있습니다.
        
        ```python
        pythonCopy code
        import dask.dataframe as dd
        
        # 예시
        ddf = dd.read_csv('your_file.csv')
        df = ddf.compute()
        
        ```
        
        1. **`Cython` 또는 `NumPy` 사용:**
            - 데이터를 CSV에서 직접 읽어와서 NumPy 배열로 처리하는 방법도 고려할 수 있습니다. 이는 특히 파일 형식이 간단하고 데이터 타입이 일정할 때 유용합니다.
        
        ```python
        pythonCopy code
        import numpy as np
        
        # 예시
        data = np.genfromtxt('your_file.csv', delimiter=',', skip_header=1)
        
        ```
        
        1. **데이터 형식 변경:**
            - CSV가 아닌 다른 형식으로 데이터를 저장하고, 더 효율적인 읽기 방법을 고려할 수 있습니다. 예를 들어, Parquet 또는 Feather와 같은 바이너리 형식은 읽기 속도가 빠를 수 있습니다.
        
        이러한 방법들 중 어떤 것이 가장 적합한지는 데이터의 크기, 형식, 사용하는 라이브러리 등에 따라 다를 수 있습니다. 실험을 통해 최적의 방법을 찾아보시기를 권장합니다.
        
    - yaml 파일을 설정 파일로 설정하기
        
        [https://otzslayer.github.io/ml/2022/03/12/write-yaml-config-in-ml-projects.html](https://otzslayer.github.io/ml/2022/03/12/write-yaml-config-in-ml-projects.html)
        
        ```jsx
        #PyYAML 설치
        
        $ pip install pyyaml
        ```
        
        ```
        # config.yaml
        
        path:
          train_path: ./data/rank.train
          valid_path: ./data/rank.test
        params:
          task: train
          objective: lambdarank
          metric: ndcg
          num_leaves: 31
          min_data_in_leaf: 50
          bagging_freq: 1
          bagging_fraction: 0.9
          min_sum_hessian_in_leaf: 5.0
          ndcg_eval_at: [1,3,5]
          learning_rate: 0.01
          num_threads: 8
        ```
        

# [1] 정보 탐색

## (1) ECG

- 심전도1
    
    [https://blog.naver.com/shinaeii/220570021088](https://blog.naver.com/shinaeii/220570021088)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2019.png)
    
    ### p파의 부재로
    
    - **AF (Atrial Fibrillation); 심방세동**
    
    미친듯한 심방의 운동으로 인해 발생되는 낮은 진폭의 파형으로,
    
    이 경우 P파 대신에 지렁이 모양의, 잔물결모양의 파형이 나타납니다.
    
    심방세동은 심전도 그래프상에서 P파의 부재와 불규칙한 QRS complex 의 특징을 보입니다.
    
- 심전도2
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2020.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2021.png)
    
    **정상인의 심전도**
    
    P : 심방의 흥분,
    
    QRS : 심실의 탈분극(수축),
    
    T : 심실의 재분극
    
    ---
    
    - P: 동방결절에서 전류 신호가 발생하고, 이것이 심방을 극성화 시키면서 판막의 심근이 수축하고[[6]](https://namu.wiki/w/%EC%8B%AC%EC%A0%84%EB%8F%84#fn-6), 심방이 비극성화 되면서 판막의 심근은 다시 이완[[7]](https://namu.wiki/w/%EC%8B%AC%EC%A0%84%EB%8F%84#fn-7)된다.방실결절이 페이스 메이킹 중이라면 P 파형이 뒤집힌다. P파의 폭은 심방의 전도시간을 의미하며 정상범위는 0.12초 이내이다. 이것이 비정상적으로 증가하면 좌심방의 확장을 시사한다. 또한 크기는 심방의 근량을 의미하며 정상범위는 심전도 검진표상 2.5mm 이내다. 역시 이것보다 크면 우심방의 확장을 의심해 볼 수 있다.
    - P-Q: 전류 신호가 심장을 자극하지 않는 휴지기라서 일직선으로 표시되며, 이때 P의 반응으로 극성화되었던 심방이 수축한다. 또한 심방이 수축할 때 심실이 반응하지 않도록 방실결절이 전류 신호를 묶어놓는다.
    - QRS Complex: 방실결절이 전류신호를 놓아주면(Q), 심실이 즉시 극성화었다(R)가 곧바로 비극성화(S)된다. 이 과정에서 심실의 판막의 심근이 수축한다.[[8]](https://namu.wiki/w/%EC%8B%AC%EC%A0%84%EB%8F%84#fn-8) (QRS 타이밍에선 아직 심실은 **박동하지 않았다.**)
    - S-T: S와 T의 사이이며 전류 신호가 심장을 자극하지 않는 휴지기라서 일직선으로 표시된다. 이때 QRS의 자극에 반응한 심실이 수축하고 심실에 있던 피가 폐와 신체 곳곳으로 출발한다.만일 일직선이 아닌 경우에는 심내/외막의 허혈 손상을 의심할 수 있다.
    - T: 심실이 다시 약하게 극성화되었다가 다시 비극성화되면서 심실과 심실 판막의 심근이 동시에 이완된다.[[9]](https://namu.wiki/w/%EC%8B%AC%EC%A0%84%EB%8F%84#fn-9) T파의 정상 범위는 사지유도일경우 5mm이내, 흉부유도일경우 10mm 이내이다. 만일 이보다 올라가면 심근의 허혈이나 고칼륨혈증을 의미하며, 평탄하거나 반대 방향일 경우도 역시 심근허혈을 의심할 수 있다.
    - U:(만약 보일 경우) 심실중격[[10]](https://namu.wiki/w/%EC%8B%AC%EC%A0%84%EB%8F%84#fn-10)(Intraventricular septum)의 재극성화로 인해 발생한다고 추정된다. 일반적으로는 보이더라도 매우 낮은 진폭을 가지고, 대다수는 아예 보이지 않는다
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2022.png)
    
- 다양한 심전도 그래프 유형 및 이해
    
    [https://quizlet.com/kr/512200082/심전도-뿌시기-flash-cards/](https://quizlet.com/kr/512200082/%EC%8B%AC%EC%A0%84%EB%8F%84-%EB%BF%8C%EC%8B%9C%EA%B8%B0-flash-cards/)
    
    [https://blog.naver.com/olololol6/222601823601](https://blog.naver.com/olololol6/222601823601)
    
    [https://jungnamgyu.tistory.com/26](https://jungnamgyu.tistory.com/26)
    
    [https://inforience.net/2019/04/28/ecg_classification/](https://inforience.net/2019/04/28/ecg_classification/)
    
    [https://www.msdmanuals.com/ko/홈/심장-및-혈관-장애/이상-심장-박동/심방-세동-및-심방-조동](https://www.msdmanuals.com/ko/%ED%99%88/%EC%8B%AC%EC%9E%A5-%EB%B0%8F-%ED%98%88%EA%B4%80-%EC%9E%A5%EC%95%A0/%EC%9D%B4%EC%83%81-%EC%8B%AC%EC%9E%A5-%EB%B0%95%EB%8F%99/%EC%8B%AC%EB%B0%A9-%EC%84%B8%EB%8F%99-%EB%B0%8F-%EC%8B%AC%EB%B0%A9-%EC%A1%B0%EB%8F%99)
    
- 심방세동1
    
    [https://www.medicalworldnews.co.kr/m/view.php?idx=1510923667#:~:text=심방세동은 부정맥 중,적절한 치료가 중요하다](https://www.medicalworldnews.co.kr/m/view.php?idx=1510923667#:~:text=%EC%8B%AC%EB%B0%A9%EC%84%B8%EB%8F%99%EC%9D%80%20%EB%B6%80%EC%A0%95%EB%A7%A5%20%EC%A4%91,%EC%A0%81%EC%A0%88%ED%95%9C%20%EC%B9%98%EB%A3%8C%EA%B0%80%20%EC%A4%91%EC%9A%94%ED%95%98%EB%8B%A4).
    
    [https://www.msdmanuals.com/ko-kr/홈/심장-및-혈관-장애/이상-심장-박동/이상-심장-박동-개요](https://www.msdmanuals.com/ko-kr/%ED%99%88/%EC%8B%AC%EC%9E%A5-%EB%B0%8F-%ED%98%88%EA%B4%80-%EC%9E%A5%EC%95%A0/%EC%9D%B4%EC%83%81-%EC%8B%AC%EC%9E%A5-%EB%B0%95%EB%8F%99/%EC%9D%B4%EC%83%81-%EC%8B%AC%EC%9E%A5-%EB%B0%95%EB%8F%99-%EA%B0%9C%EC%9A%94)
    
    [https://www.msdmanuals.com/ko-kr/홈/심장-및-혈관-장애/이상-심장-박동/심방-세동-및-심방-조동](https://www.msdmanuals.com/ko-kr/%ED%99%88/%EC%8B%AC%EC%9E%A5-%EB%B0%8F-%ED%98%88%EA%B4%80-%EC%9E%A5%EC%95%A0/%EC%9D%B4%EC%83%81-%EC%8B%AC%EC%9E%A5-%EB%B0%95%EB%8F%99/%EC%8B%AC%EB%B0%A9-%EC%84%B8%EB%8F%99-%EB%B0%8F-%EC%8B%AC%EB%B0%A9-%EC%A1%B0%EB%8F%99)
    
    - 심방 세동이나 심방 조동의 경우, 심방은 각 맥박시 혈액이 심실로 완전히 이동하지 않아 심방이 비워지지 않습니다. 시간이 경과하면서, 심방 내 일부 혈액이 침체되어 혈전이 생성될 수 있습니다.
    - 심방 세동이나 심방 조동으로 인해 심장이 지나치게 급격히 박동하게 되면, 심실이 혈액으로 완전히 채워지기까지의 시간이 충분하지 못합니다. 완전히 채워지지 않으면 심장에 공급되는 혈액량이 감소하게 됩니다. 이러한 혈액량의 감소는 혈압을 떨어트리고 [심부전](https://www.msdmanuals.com/ko-kr/%ED%99%88/%EC%8B%AC%EC%9E%A5-%EB%B0%8F-%ED%98%88%EA%B4%80-%EC%9E%A5%EC%95%A0/%EC%8B%AC%EB%B6%80%EC%A0%84/%EC%8B%AC%EB%B6%80%EC%A0%84-hf)의 원인이 될 수도 있습니다.
    
    - 심방갈 때 p파, 심실에서 qrs파, 심실 회복기에 t파
    
- 심방세동2
    
    심방세동은 심방 내에서 발생하는 빠르고 불규칙한 전기 신호로 가장 흔한 부정맥의 한 종류입니다. 약 2%의 유병률을 보이며, 30초 이상 지속되는 심방세동이 7일 이내 자발적으로 사라지는 경우 발작성 심방세동이라 하며, 7일 이상 지속되는 경우 **지속성 심방세동**이라 합니다.
    
    - **심방세동 증상**
    
    심방세동의 증상과 정도는 다양합니다. **심방세동 환자의 약 25%는 무증상이며, 가장 흔한 증상은 두근거림, 피로, 호흡곤란, 현기증이며, 다뇨증과 실신도 발생할 수 있습니다.** 뇌졸중과 같은 혈전색전증이 합병증으로 나타날 수 있으며 결국 울혈성 심부전으로 진행하게 됩니다. 신체검진에서 특징적으로 불규칙한 맥박이 나타납니다.
    
    - **심방세동의 원인**
    
    심방세동은 심장에 특별한 구조적 이상 없이 노화와 관련되어 발생 빈도가 증가하는 것으로 알려져 있습니다. 그 외에 고혈압, 허혈성심장질환, 승모판질환, 비후성 심근증, 확장성 심근증 등의 심장질환과 관련이 있습니다.
    
    폐색성 수면 무호흡증은 저산소증, 자율신경계 흥분, 고혈압을 유발하여 심방세동의 위험을 증가시킵니다. 비만은 심외막에 지방 축적을 유발하여 심방세동의 위험을 증가시킵니다.
    
    또한 심방세동을 유발하는 일시적이고 가역적인 원인이 있을 수 있습니다. 가장 흔한 원인은 과도한 알코올 섭취, 개방형 심장 및 흉부 수술, 급성심근경색, 심낭염, 심근염, 폐색전증 등입니다. 갑상선기능항진증과 같은 전신질환도 교정 가능한 심방세동의 원인입니다.
    
    - **심방세동의 진단(검사)**
    
    심방세동을 진단하기 위한 가장 정확한 검사 방법은 심전도입니다. 심전도에서 불규칙한 맥박 및 불규칙한 이소성 P파와 QRS군이 관찰됩니다. 발작성 심방세동의 경우 보행 모니터링이 유용하며, 증상이 매일 발생하는 경우 24시간 홀터검사가 유용합니다.
    
    갑상선기능항진증 혹은 과도한 알코올 섭취와 같이 교정 가능한 원인을 찾고, 구조적 심장질환 및 동반 질환을 감별하기 위해 추가 검사가 필요할 수 있습니다. 갑상선 기능검사, 간 기능검사, 신장 기능검사 등을 시행하고, 심장 크기와 좌심실의 기능 평가, 선천성 심장질환 및 판막 질환을 확인하기 위해 심장 초음파 검사를 진행합니다. 병력이나 신체검사에서 폐질환이 의심되는 경우에는 흉부 엑스레이 검사를 시행합니다. 허혈성 심장질환을 평가하기 위해 운동부하검사, 관상동맥조영술 등이 필요할 수 있습니다.
    
    - **심방세동의 치료**
    
    심방세동을 정상 심박동으로 전환시키기 위한 치료로는 항부정맥 약물, 전기적 심율동 전환, 고주파 전극도자 절제술, 냉각풍선 절제술 등의 방법이 있습니다. 그리고 심방세동의 중대한 합병증인 뇌졸중이나 전신 혈전색전증을 예방하기 위해 항응고제를 사용합니다.
    
    우선 항부정맥 약물치료를 해 보고 약물에 반응이 적거나 부작용으로 인해 약물복용을 지속하기 어려운 환자를 대상으로 **고주파 전극도자 절제술** 또는 **냉각풍선 절제술**이라는 시술적 치료를 시행할 수 있습니다. 시술적 치료란 두꺼운 다리 혈관을 통해 관을 삽입해서 심장까지 도달한 다음, 비정상적인 전기신호가 나오는 심장 부위를 고주파 열이나 냉동 에너지를 사용해 절제하고 전기적으로 격리시키는 방법입니다.
    
    심방세동 환자는 건강한 사람에 비해 뇌졸중 발생 위험이 5배 정도 높으며, 심장병으로 인한 사망률도 2배 정도 높아지므로 의사의 처방과 적절한 관리가 매우 중요합니다.
    
- 심전도 리드
    
    **2개 전극 쓰는 방법 (표준사지유도)**
    
    ( + ) 전극 하나 ( - ) 전극 하나
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2023.png)
    
    Lead1: 오른팔(-) => 왼팔 (+)
    
    Lead2: 오른팔(-) => 왼쪽 다리(+)
    
    Lead3: 왼팔(-) => 왼쪽다리(+)
    
    **3개 전극 쓰는 방법 (단극사지유도)**
    
    ( + ) 전극 하나 ( - ) 전극 2개
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2024.png)
    
    전기적 신호가 양극으로 다가오면 양성 편향이 나타납니다.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2025.png)
    
    전기적 신호가 양극에서 멀어지면 음성 편향이 나타납니다.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2026.png)
    
    전기적 신호가 양극과 음극을 이은 선을 지나면 이중편향이 나타납니다.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2027.png)
    
    ---
    
    심근이 흥분하면서 양성 편향이 기록되는 모식도 입니다.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2028.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2029.png)
    
    *Lead2*
    
    : 오른팔(-) => 왼쪽 다리(+) // 우리가 일반적으로 처음 접하는 심전도가 나타납니다. 심장 내부에서 전기자극이 흐르는 방향과 같이 ↘ 방향으로 전극이 붙어있어 심전도 그래프를 가장 보기 좋습니다.
    

## ****(2)**** 논문 및 기타 링크

- 논문
    - cnn+lstm 가벼운 모델:  **A lightweight hybrid CNN-LSTM model for ECG-based arrhythmia detection**
        
        [https://arxiv.org/pdf/2209.00988.pdf](https://arxiv.org/pdf/2209.00988.pdf)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2030.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2031.png)
        
        ---
        
        - 구현
            
            ```python
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            class CNNLSTMModel(nn.Module):
                def __init__(self):
                    super(CNNLSTMModel, self).__init__()
                    
                    # 첫번째 Convolutional Block
                    self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50, stride=1)
                    self.relu1 = nn.ReLU()
                    self.maxpool1 = nn.MaxPool1d(kernel_size=20, stride=2)
                    self.dropout1 = nn.Dropout(p=0.1)
            
                    # 두번째 Convolutional Block
                    self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=10, stride=1)
                    self.relu2 = nn.ReLU()
                    self.maxpool2 = nn.MaxPool1d(kernel_size=10, stride=2)
                    self.dropout2 = nn.Dropout(p=0.1)
            
                    # 세번째 Convolutional Block
                    self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1)
                    self.relu3 = nn.ReLU()
                    self.maxpool3 = nn.MaxPool1d(kernel_size=5, stride=2)
                    self.dropout3 = nn.Dropout(p=0.1)
            
                    # LSTM Block
                    self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
                    self.dropout_lstm = nn.Dropout(p=0.1)
            
                    # Dense Layers
                    self.dense1 = nn.Linear(32, 32)
                    self.relu_dense1 = nn.ReLU()
                    self.dropout_dense1 = nn.Dropout(p=0.1)
            
                    self.dense2 = nn.Linear(32, 16)
                    self.relu_dense2 = nn.ReLU()
            
                    # Output Layer - 이중분류를 위해 출력 유닛을 1로 설정
                    self.output = nn.Linear(16, 1)
            
                def forward(self, x):
                    # x: ECG 데이터 (Batch Size, Channels, Length)
                    
                    # Convolutional Blocks
                    x = self.dropout1(self.maxpool1(self.relu1(self.conv1(x))))
                    x = self.dropout2(self.maxpool2(self.relu2(self.conv2(x))))
                    x = self.dropout3(self.maxpool3(self.relu3(self.conv3(x))))
            
                    # LSTM Layer - LSTM은 추가적인 차원을 요구하기 때문에 차원 조정이 필요합니다.
                    x = x.permute(0, 2, 1) # (Batch Size, Sequence Length, Features)
                    x, (hn, cn) = self.lstm(x)
                    x = self.dropout_lstm(x[:, -1, :]) # 마지막 시퀀스의 출력만 사용
            
                    # Dense Layers
                    x = self.dropout_dense1(self.relu_dense1(self.dense1(x)))
                    x = self.relu_dense2(self.dense2(x))
            
                    # Output Layer - 시그모이드 활성화 함수를 사용하여 0과 1 사이의 값을 출력
                    x = torch.sigmoid(self.output(x))
                    return x
            
            # 모델 인스턴스화 및 요약
            model = CNNLSTMModel()
            print(model)
            ```
            
        
        ### **논문 요약**
        
        - **목적**: 심전도(ECG) 신호를 이용하여 심장 부정맥을 자동으로 분류하고 식별하는 컴퓨터화된 방법 개발.
        - **방법론**: 500개의 ECG 샘플 세그먼트를 사용하여 11층 네트워크를 통한 리듬 분류를 수행. 이 모델은 수동 특징 추출 없이 end-to-end 방식으로 작동합니다.
        - **데이터셋**: MIT-BIH 부정맥 데이터베이스와 장기 AF 데이터베이스에서 ECG 신호를 선택.
        - **모델 구조**: CNN과 LSTM을 결합한 구조로, 11개의 층으로 구성됨.
        - **성능**: 평균 진단 정확도 98.24% 달성.
        - **손실 함수 (Loss Function)**: 논문에서는 **`Categorical Cross-Entropy`**를 손실 함수로 사용했습니다. 이는 다중 클래스 분류 문제에 주로 사용되는 손실 함수입니다.
        - **옵티마이저 (Optimizer)**: 모델을 최적화하기 위해 **`Adam Optimizer`**를 사용했습니다. Adam 옵티마이저는 학습률을 자동으로 조정하면서 효율적인 최적화를 가능하게 하는 알고리즘입니다.
        
        ---
        
        ### **모델 구현 단계**
        
        1. **데이터 준비**: MIT-BIH 부정맥 데이터베이스와 장기 AF 데이터베이스에서 ECG 데이터를 수집합니다.
            1. 심전도 데이터를 기반으로 심방세동 진단에 대한 프레임워크를 제안하는 데에 MIT-BIH 부정맥 데이터베이스와 장기심방세동(LTAF) 데이터베이스를 활용하였습니다. MIT-BIH 부정맥 데이터베이스는 47명의 환자로부터 얻은 48개의 반시간 동안의 ECG 레코딩으로 구성되었으며, 전통적인 12개의 리드 대신 MLII와 V1 리드로 기록되었습니다. LTAF 데이터베이스에는 84명의 환자로부터 얻은 24시간 동안의 ECG 신호가 포함되어 있으며, 이러한 신호는 128Hz 샘플링 주파수로 동기화하여 기록되었습니다.
        2. **전처리**: 노이즈 필터링과 데이터 리샘플링을 수행합니다.
            1. 노이즈 필터링은 ECG 신호에 영향을 미치는 다양한 유형의 노이즈를 제거하기 위해 적용되었습니다. 또한 MIT-BIH ECG 레코딩은 360Hz로 샘플링되었으며, LTAF 데이터베이스의 ECG 레코딩은 128Hz로 샘플링되었습니다. 따라서 두 데이터베이스를 모두 사용하려면 MIT-BIH 데이터셋의 신호를 다운샘플링해야 했습니다. 이로써 모든 레코드의 주파수는 128Hz로 일치시켰습니다.
        3. **모델 아키텍처 구성**:
            - CNN 레이어: 특징 추출을 위해 1차원 컨볼루션 레이어 사용.
            - LSTM 레이어: 시퀀스 데이터의 장기 의존성을 처리하기 위해 사용.
            - 드롭아웃 레이어: 과적합 방지를 위해 사용.
            - 완전 연결 레이어: 특징을 최종 출력 벡터로 변환.
        4. **모델 훈련**: 85%의 데이터를 사용하여 모델을 훈련합니다.
        5. **성능 평가**: 나머지 15%의 데이터를 사용하여 모델을 평가합니다.
        
        ---
        
        ECG 레코딩의 분할은 모델에 입력될 데이터의 길이를 균일하게 만드는 과정입니다. 이 연구에서는 128Hz의 샘플링 속도와 평균 심장 주기 0.8초를 고려하여 500개 샘플 (3.9초)로 된 세그먼트가 추출되었습니다. 세그먼트는 겹치는 방식으로 추출되었으며 모든 ECG 세그먼트가 합쳐졌습니다. 정상 및 심방세동 클래스와 관련된 세그먼트가 과도하게 높았기 때문에 이러한 불균형의 부작용을 제거하기 위해 교육 및 테스트 단계에서 평가 지표가 각 클래스의 크기의 역수로 가중화되었습니다.
        
        이 논문은 CNN과 LSTM 아키텍처의 조합을 기반으로 하는 고정확도 딥 러닝 기술을 소개하여 원시 ECG 신호에서 다양한 유형의 심장 부정맥을 진단하려고 합니다. 
        
        이 모델은 11개의 레이어로 구성되었으며, 데이터의 85%를 사용하여 교육되었으며 나머지 15%의 세그먼트 데이터를 사용하여 평가되었습니다. 
        
        이 모델은 1차원 합성곱 레이어와 ReLu (Rectified Linear Unit) 활성화 함수를 포함하는 세 개의 연속된 컨볼루션 블록으로 시작하며, Maxpooling 레이어를 통해 주요 기능을 선택하고 특성 차원을 줄입니다. 
        
        마지막 컨볼루션 블록의 출력은 LSTM 레이어로 이어집니다. LSTM은 기억 블록으로 구성되어 있으며 장기 의존성과 폭발적인 그래디언트 문제를 다룰 수 있습니다. 또한 과적합 문제를 방지하기 위해 드롭아웃 레이어가 사용되었습니다. 이 모델에서는 몇 개의 밀집 레이어가 사용되어 벡터 차원을 변경하고 몇 차원의 특성을 선형 벡터로 변환합니다. 마지막 레이어로 다중 클래스 활성화 함수인 소프트맥스가 사용되어 클래스의 출력 벡터를 각 클래스에 대한 확률로 변환합니다. 따라서 범주형 타겟 변수를 다중 분류 과정에서 원핫 인코딩 프로세스를 사용하였습니다. 이 연구에서 구현된 CNN 모델과 LSTM 레이어에 대한 자세한 설명은 표 II에서 확인할 수 있습니다.
        
        ---
        
        IV. 결과 및 토의
        
        이 연구의 목적은 8가지 부정맥 및 정상 심장 리듬을 분류하는 것입니다. 따라서 데이터 준비 후 데이터셋을 교육 및 테스트 두 그룹으로 나누었습니다. 실험 결과 교육 데이터의 크기가 클수록 테스트 데이터에서의 기능이 향상되므로 준비된 데이터의 85%가 제안된 모델을 교육하고 15%의 데이터가 모델을 테스트하고 검증하는 데 사용되었습니다. 모델은 교육 및 테스트 세트에서 100번의 에포크로 컴파일되었습니다. 손실 함수로 범주형 크로스 엔트로피를 사용하고 Adam 옵티마이저가 선택되었으며 이는 알고리즘 수렴을 빠르게 만들며 구현이 간단하며 계산 효율적이며 기본 하이퍼파라미터를 약간 조정하여 사용할 수 있기 때문입니다. 가중치와 바이어스는 손실 함수에 의해 생성된 값을 활용하여 원하는 최적화된 값을 얻을 때까지 지속적으로 업데이트됩니다.
        
        A. 성능 평가 지표
        
        다중 클래스 분류 모델의 성능을 평가하기 위한 여러 가지 다른 메트릭스가 있습니다. 이 연구에서 사용된 세 가지 메트릭스는 정확도 (Acc), 민감도 (Se), 및 특이도 (Sp)입니다.
        
        - 정확도는 모델의 성능을 얼마나 정확하게 평가하는지 나타냅니다.
        - 민감도는 각 클래스의 진짜 양성을 예측하는 모델의 능력을 평가하는 메트릭스입니다.
        - 특이도는 각 클래스의 진짜 음성을 예측하는 모델의 능력을 평가합니다.
        
        B. 실험 결과 및 토의
        
        그림 2는 분류 알고리즘의 성능을 시각화하는 데 사용되는 혼동 행렬을 보여줍니다. 이 행렬의 주 대각선에 있는 숫자는 진짜 양성을 나타냅니다. 민감도와 특이도 매개 변수는 성능을 평가하기 위해 사용됩니다. 그림 3에서 제안된 모델이 특정 리듬이 고려될 때 다른 리듬을 올바르게 식별하는 능력 (특이도)이 8가지 모든 부정맥 및 정상 리듬에 대해 90% 이상입니다.
        
        정확도는 모델이 얼마나 정확하게 수행했는지 나타냅니다. 제안된 모델이 달성한 평균 분류 정확도는 98.24%입니다. 이 연구에서 구현된 1D-CNN+LSTM 모델은 다양한 부정맥 유형을 분류하는 데 높은 성능의 정확도를 나타냈습니다. 그 구현은 직관적이며 SVM 분류기 기반 전략, 무작위 포레스트 알고리즘 [14] 또는 SVM 기반 메서드와 앙상블 분류기의 배치와 같은 대부분의 최신 접근 방법과 비교하여 낮은 계산 복잡성을 가지고 있습니다. 또한 제안된 모델은 상대적으로 낮은 수의 레이어로 이루어져 있으며 [11], [15], [17], [18]에서 실행된 모델과는 달리 적은 수의 레이어를 가지고 있습니다. 이전 연구에서는 주로 ECG 신호의 단일 데이터베이스만 사용되었습니다 [3], [16], 하지만 이 연구에서는 서로 다른 샘플링 주파수를 가진 두 데이터베이스의 조합을 교육 및 테스트 절차에 사용하였습니다. 또한 이전 연구 중 대부분은 [13], [16]과 같이 제한된 수의 부정맥 질환만 분류했지만 이 시스템에서는 9가지 다른 심장 리듬 유형을 구별하였습니다. 이와 관련된 성능 비교는 표 III에 제시되어 있습니다.
        
        ROC 곡선은 분류기 진단 능력을 그래픽으로 나타내는 평가 메트릭스입니다. 다양한 임계값 설정에서 진짜 양성률 (TPR)도 알려진 민감도와 거짓 양성률 (FPR)도 알려진 (1-특이도)를 플로팅하여 생성됩니다. 곡선 아래 영역 (AUC)은 분류기가 클래스 간을 얼마나 잘 구별했는지를 나타내는 측정값입니다. 그림 4에서 볼 수 있듯이, 모델은 각 부정맥 클래스를 구별하는 데 거의 완벽한 AUC를 달성했습니다.
        
        제안된 1D-CNN+LSTM 모델의 높은 계산 복잡도에 직면한 경우 CNN은 실시간 및 전력 효율적인 시나리오에서의 더 넓은 적용을 위한 중요한 도전 과제를 제시합니다. 다음으로, 모델이 가벼우며 홀터 모니터 장치에서 사용될 수 있다는 것을 입증하기 위해 모델 크기와 단일 리듬을 분류하는 데 걸리는 추론 시간을 보여드리겠습니다. 제안된 모델의 Raspberry Pi에서의 추론 시간은 단 한 번의 리듬을 분류하는 데 5.127 ms로, 한 리듬을 분류하는 데 5.127 ms가 걸린다는 의미입니다. Raspberry Pi(홀터 모니터에 사용되는 프로세서)에 로드되는 모델의 크기는 0.16 MB입니다. 이것은 우리의 모델이 Raspberry Pi 기반의 홀터 모니터 장치에서 경쟁력 있는 것을 의미합니다.
        
        이 연구는 다양한 부정맥 유형을 식별하고 정확하게 분류할 수 있는 높은 정확도를 가진 1D-CNN+LSTM 모델을 제안하였습니다. 또한 이 모델은 가벼워서 실시간 및 저전력 환경에서 사용 가능하며 Raspberry Pi와 같은 장치에서 효율적으로 실행될 수 있습니다. 이 모델은 다양한 부정맥을 구분하고 실용적인 응용 프로그램에 사용할 수 있는 중요한 도구로서의 잠재력을 가지고 있습니다.
        
        ---
        
        V. 결론 및 향후 연구
        
        심장 부정맥의 정확한 감지는 환자의 조기 치료에 중요하며 컴퓨터 지원 진단은 중요한 역할을 할 수 있습니다. 본 논문에서는 MIT-BIH 및 Long-term AF 두 데이터베이스에서 얻은 ECG 레코딩을 사용하여 실험을 진행했습니다. 제안된 CNN+LSTM 모델은 8가지 다른 부정맥 및 정상 ECG 신호의 분류를 수행할 수 있습니다. 이 모델은 CNN을 통해 심장 신호의 판별 특성과 정보를 추출하고 LSTM 레이어를 통해 시계열적인 특성을 얻을 수 있습니다. 실험 결과, 보이지 않은 단일 리듬에 대한 추론 시간이 5.127 ms로 평균 테스트 정확도는 98.24%에 달했습니다.
        
        부정맥을 감지하기 위해 심박동의 형태를 사용하는 것 외에도 모델을 교육하기 위해 신호의 다른 속성의 변화를 활용하는 것이 더 나은 성능을 달성하는 데 도움이 될 수 있습니다. 이러한 속성 중 일부는 RR 간격 및 QRS 간격(주기) 등입니다. 이 모델에서는 더 높은 부정맥 감지 정확도를 얻기 위해 합성곱 신경망 뒤에 LSTM 레이어가 따르도록 설계되었습니다. 그러나 일부 부정맥의 분류 정확도를 개선해야 합니다. 이 모델과 함께 규칙 기반 알고리즘을 배치하면 더 나은 성능을 얻을 수 있습니다. 이전 연구 대부분은 주로 하나의 데이터베이스(주로 MIT-BIH)에서 수행되었지만 이 연구는 두 데이터베이스의 조합을 기반으로 합니다. 그러나 더 신뢰할 수 있는 결과를 얻기 위해 더 많은 데이터베이스의 조합이 필요합니다. 이 연구에서는 모델을 교육하고 테스트하기 위해 두 리드(일반적으로 MLII 및 V1 리드)에서 얻은 ECG 신호를 사용했지만 임상 응용에서는 12-리드 ECG가 표준으로 알려져 있습니다. 이상적인 모델은 표준 ECG 신호와 구별할 수 있어야 합니다. 이 모델은 8가지 다양한 심장 부정맥과 정상 동맥 리듬을 높은 정확도로 분류할 수 있었지만 여전히 중요한 감지되지 않은 부정맥 및 심장 질환이 있습니다. 구현된 모델은 더 많은 부정맥을 구별할 수 있도록 개선되어야 합니다.
        
    - 1d cnn 논문 리뷰 (auc 99%) : **Detection of Atrial Fibrillation Using 1D Convolutional Neural Network**
        
        [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180882/pdf/sensors-20-02136.pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180882/pdf/sensors-20-02136.pdf)
        
        # **2.1. 시스템 개요**
        
        - **데이터 길이 정규화, 오프라인 훈련, 온라인 예측**을 포함하는 AF 탐지 시스템을 제안합니다.
        - **데이터셋**: AF, 정상, 잡음, 기타의 네 가지 클래스를 포함하는 ECG 신호.
        - **1D CNN**: ECG 신호 분류를 위해 설계된 1차원 컨볼루션 신경망.
        - **훈련 단계**: 네트워크의 각 층에 대해 오류를 역전파하여 반복적으로 네트워크 매개변수를 업데이트합니다.
        - **추론 단계**: 최적화된 모델을 사용하여 테스트 데이터에 대한 예측 결과를 생성합니다.
        
        ---
        
        # **2.2. 데이터 길이 정규화**
        
        - **길이 정규화 알고리즘**: 히스토그램 기반 길이 정규화 알고리즘을 개발하여 자동으로 길이 임계값을 결정합니다.
        - **임계값**: 30초, 즉 9000 샘플 (샘플링 속도 300 샘플/초).
        - **데이터 처리**: 9000 샘플보다 긴 녹음은 50% 중첩을 가진 여러 세그먼트로 생성되며, 9000 샘플 미만인 경우 동일한 레이블을 가진 녹음을 연결합니다.
        
        ### **리뷰**
        
        - **데이터 길이 정규화**: ECG 데이터의 가변 길이 문제를 해결하는 효과적인 접근 방식입니다. 히스토그램 기반 알고리즘은 데이터의 특성을 고려하여 임계값을 설정함으로써 편향을 최소화합니다.
        - **1D CNN 구조**: ECG 신호 분류를 위한 1D CNN의 사용은 시계열 데이터 처리에 적합합니다. 네트워크의 훈련과 추론 단계는 명확하게 설명되어 있으며, 실제 ECG 데이터에 대한 적용 가능성이 높아 보입니다.
        - **효율적인 데이터 활용**: 녹음 길이가 9000 샘플보다 긴 경우 여러 세그먼트로 나누어 처리하는 방식은 데이터 활용을 극대화합니다. 이는 특히 제한된 데이터셋에서 유용할 수 있습니다.
        - **실제 적용 가능성**: 제안된 시스템은 실제 의료 환경에서 AF 탐지를 위한 실용적인 솔루션을 제공할 수 있습니다. 특히, 오프라인 훈련과 온라인 예측의 조합은 실시간 모니터링 및 진단에 유용할 것으로 보입니다.
        
        종합적으로, 이 논문은 ECG 데이터를 처리하고 심방세동을 탐지하기 위한 효과적인 딥러닝 기반 방법론을 제시합니다. 데이터 전처리와 네트워크 설계 모두 실제 의료 데이터에 적용될 수 있는 실용적인 접근 방식을 보여줍니다.
        
        ---
        
        ### 데이터 길이 정규화
        
        1. 만약 녹음의 길이가 9000 샘플보다 크다면,
        2. 녹음을 9000 샘플로 잘라내고, 세그먼트 간에 50% 중첩을 둡니다.
        3. 만약 녹음의 길이가 9000 샘플보다 작다면,
        4. 데이터: 녹음을 복사합니다.
        5. 복사한 데이터를 녹음의 뒤에 붙입니다.
        6. 붙인 녹음이 9000 샘플에 도달할 때까지 5단계를 반복합니다.
        7. 만약 녹음의 길이가 9000 샘플과 같다면,
        8. 녹음을 그대로 유지합니다.
        
        이 의사코드는 ECG 데이터의 길이를 표준화하는 과정을 명확하게 설명합니다. 9000 샘플이라는 고정된 길이를 설정함으로써, 다양한 길이의 ECG 녹음을 효과적으로 처리할 수 있습니다. 이러한 접근 방식은 특히 딥러닝 모델에 입력되는 데이터의 일관성을 유지하는 데 중요합니다. 또한, 이 방법은 데이터의 양을 증가시켜 모델의 학습에 도움을 줄 수 있으며, 특히 데이터가 부족한 경우에 유용할 수 있습니다.
        
        ### "세그먼트 간에 50% 중첩을 둔다"는 말은,
        
         데이터를 세그먼트로 나눌 때 각 세그먼트가 이전 세그먼트의 절반을 포함하도록 한다는 의미입니다. 이를 통해 데이터의 연속성을 유지하면서도 중복을 허용하여 더 많은 학습 데이터를 생성할 수 있습니다.
        
        예를 들어, 녹음이 18000 샘플이라고 가정해 보겠습니다. 이 경우, 9000 샘플의 길이를 가진 세그먼트를 만들 때, 첫 번째 세그먼트는 1번째 샘플부터 9000번째 샘플까지를 포함하고, 두 번째 세그먼트는 4501번째 샘플부터 13500번째 샘플까지를 포함합니다. 이렇게 하면 두 세그먼트는 4500 샘플(9000의 50%)을 공유하게 됩니다. 이런 방식으로 데이터를 세그먼트로 나누면, 각 세그먼트가 이전 세그먼트의 일부를 포함하게 되어 데이터의 중복을 허용하면서도 다양한 데이터를 확보할 수 있습니다.
        
        ---
        
        # 2.3.1 'CNN Architecture'
        
        ### 딥 뉴럴 네트워크 아키텍처 설계의 도전
        
        - 딥 뉴럴 네트워크 아키텍처 설계는 성능 지표, 손실 함수, 최적화 알고리즘, 하이퍼파라미터 설정 등 여러 측면을 고려해야 합니다.
        - 하이퍼파라미터 설정에는 은닉층의 수, 뉴런 수, 각 층의 채널 수, 학습률, 배치 크기, 배치 정규화, 풀링 등이 포함됩니다.
        
        ### 전이 학습과 1D 네트워크
        
        - 2D 이미지에서는 전이 학습이 널리 사용되며, LeNet, AlexNet, VGG, Inception, ResNet과 같은 우수한 사전 훈련된 2D 네트워크가 있습니다.
        - 하지만 1차원 데이터에 대한 사전 훈련된 네트워크는 매우 드물기 때문에, 네트워크를 처음부터 설계해야 합니다.
        
        ### 제안된 아키텍처
        
        - **제안된 아키텍처는 10개의 컨볼루션 블록, 2개의 완전 연결층, 그리고 출력 예측을 위한 소프트맥스 층을 포함합니다.**
        - **컨볼루션 블록은 컨볼루션 층, ReLU 층, 맥스풀링 층으로 구성**됩니다.
        - **첫 번째 컨볼루션 블록에만 ReLU 활성화 후 배치 정규화(BN) 층을 추가하여 입력층을 정규화**합니다.
        - 이후 **5개의 컨볼루션 블록에 드롭아웃 층을 추가**합니다.
        - **마지막 블록에서는 컨볼루션 층과 ReLU 층만 적용**합니다.
        
        ### 필터 크기와 커널 크기
        
        - **컨볼루션 층의 필터 크기는 32로 시작하여, 2개 블록마다 2배씩 증가**합니다.
        - **모든 층의 커널 크기는 계산 부하를 줄이기 위해 5로 설정**됩니다.
        
        ### 풀링과 소프트맥스 함수
        
        - **각 컨볼루션 출력은 커널 크기가 2인 풀링 층을 통해 다운샘플링**됩니다.
        - **마지막 층에서는 소프트맥스 함수를 사용하여 네 출력 클래스에 대한 예측 확률을 생성**합니다.
        
        ### 레이블 인코딩
        
        - 실제 레이블은 **원-핫 인코딩 벡터로 변환**됩니다.
        - 예를 들어, **"AF" 레이블은 [1, 0, 0, 0]의 원-핫 인코딩 벡터로 표현**됩니다.
        
        ### 리뷰
        
        이 논문의 CNN 아키텍처 설계는 심방세동(AF) 탐지를 위한 ECG 데이터 처리에 매우 적합한 접근 방식을 보여줍니다. 특히, 1차원 데이터에 대한 사전 훈련된 네트워크가 드문 상황에서 네트워크를 처음부터 설계한 점은 주목할 만합니다. 10개의 컨볼루션 블록과 2개의 완전 연결층을 포함하는 이 아키텍처는 ECG 데이터의 특성을 효과적으로 학습할 수 있을 것으로 보입니다. 또한, 배치 정규화와 드롭아웃 층의 적절한 사용은 모델의 일반화 성능을 향상시킬 수 있습니다. 필터 크기와 커널 크기의 설정은 계산 효율성과 성능 사이의 균형을 잘 고려한 것으로 보입니다. 전반적으로, 이 아키텍처는 ECG 데이터를 기반으로 한 심장 질환 탐지에 있어서 효과적인 딥러닝 모델을 제공할 것으로 기대됩니다.
        
        ---
        
        ## 3.1. 데이터셋
        
        - **데이터셋**: PhysioNet Challenge 2017에서 제공한 8528개의 단일 리드 가변 길이 ECG 녹음(9초에서 61초 사이).
        - **데이터 처리**: 300Hz에서 샘플링되고 AliveCor KardiaMobile 장치에 의해 밴드패스 필터링됨.
        - **클래스 구성**: AF(심방세동) 771개, 정상 5154개, 잡음 46개, 기타 2557개.
        - **데이터 길이 정규화 후**: 녹음 수가 10,151개로 증가.
        
        ## 3.2. 평가 지표
        
        - **F1 점수**: 입력 ECG 기록을 네 가지 클래스 중 하나로 예측하는 시스템의 성능을 평가하기 위해 사용.
        - **정밀도와 재현율**: 혼동 행렬에서 유도되며, 각 행은 예측된 클래스를, 각 열은 실제 클래스를 나타냄.
        - **클래스별 F1 점수 계산**: 4×4 크기의 혼동 행렬을 2×2 크기로 변환하여 계산.
        - **정확도 대신 F1 점수 사용**: 불균형 데이터셋에서 정확도만으로는 부족하기 때문에 F1 점수 사용.
        
        ## 3.3. K-폴드 교차 검증
        
        - **K-폴드 교차 검증**: 머신러닝에서 가장 인기 있는 방법 중 하나.
        - **층화 K-폴드 교차 검증**: 불균형 분류에 특히 우수한 층화 샘플링 버전의 K-폴드.
        - **데이터셋 분포**: 10,151개 세그먼트로 구성된 불균형 데이터셋.
        - **최적의 K 값 찾기**: 80:20의 훈련/테스트 비율이 최상의 성능을 달성, K = 5로 결정.
        
        ## 3.4. 하이퍼파라미터 최적화
        
        - **그리드 탐색 알고리즘**: 층 수, 커널 크기, 배치 크기, 학습률에 대한 하이퍼파라미터 최적화에 사용.
        - **학습률 설정**: Adam 최적화기의 제안된 학습률 0.001을 기준으로 설정.
        - **배치 크기, 커널 크기, 층 수 선택**: 다양한 조합을 통해 300개의 신경망 아키텍처 생성.
        - **최적 아키텍처 탐색**: 그리드 탐색을 통해 최적의 아키텍처를 찾고, GPU를 사용하여 훈련 속도 향상.
        
        ### 리뷰
        
        이 논문의 수치 분석 부분은 ECG 데이터를 기반으로 한 심장 질환 탐지에 대한 심도 있는 분석을 제공합니다. 데이터셋의 상세한 설명과 데이터 길이 정규화 과정은 모델의 훈련에 있어 중요한 기초를 마련합니다. F1 점수를 사용한 성능 평가 방법은 불균형 데이터셋에 대한 정확한 평가를 가능하게 합니다. K-폴드 교차 검증과 하이퍼파라미터 최적화 접근 방식은 모델의 일반화 성능을 향상시키는 데 중요한 역할을 합니다. 전반적으로, 이 논문은 ECG 데이터를 활용한 심장 질환 탐지를 위한 신뢰할 수 있는 방법론을 제시하며, 특히 데이터 전처리와 모델 최적화에 대한 실용적인 접근 방식을 강조합니다.
        
        ---
        
        ## **3.5.1. 예측 정확도**
        
        - **레이블 할당**: 각 녹음은 AF, 정상, 잡음, 기타 중 하나로 분류됩니다.
        - **1D CNN 모델**: 테스트 세트에 대한 레이블을 추론합니다.
        - **그리드 탐색 결과**: 최적의 모델을 통해 네 클래스의 평균 F1 점수를 평가합니다.
        - **하이퍼파라미터 조합**: 컨볼루션 층 수, 필터 커널 크기, 배치 크기, 학습률 등을 포함합니다.
        - **최적 조합**: 층 수 = 10, 커널 크기 = 5, 학습률 = 0.0001, 배치 크기 = 30으로 평균 F1 점수 77.8% 달성.
        
        ## **3.5.2. 네트워크 아키텍처 분석**
        
        - **배치 정규화와 맥스풀링**: 훈련 가속화에 널리 사용되는 기술입니다.
        - **배치 정규화 실험**: 모든 컨볼루션 층 후에 배치 정규화를 추가하는 것은 성능 저하를 초래합니다.
        - **맥스풀링 실험**: 맥스풀링을 평균 풀링으로 대체한 Proposed-2가 Proposed-1 및 그 변형보다 우수한 성능을 보입니다.
        - **평균 풀링의 효과**: 이전 층에서의 대부분의 정보를 유지하고, 이를 계층별로 전달하는 데 유리합니다.
        
        ### **리뷰**
        
        이 논문의 결과 및 분석 부분은 1D CNN 모델을 사용하여 ECG 데이터를 분류하는 데 있어 중요한 통찰을 제공합니다. 특히, 다양한 하이퍼파라미터 조합에 대한 실험은 최적의 모델 구성을 찾는 데 중요한 역할을 합니다. 
        
        **배치 정규화와 맥스풀링에 대한 실험**은 이러한 기술들이 네트워크 성능에 미치는 영향을 잘 보여줍니다. 배치 정규화를 네트워크의 모든 층에 추가하는 것이 항상 이점이 되지 않음을 보여주며, 맥스풀링 대신 평균 풀링을 사용하는 것이 더 나은 결과를 가져올 수 있음을 시사합니다.
        
        ---
        
        ## 3.5.3. 네트워크 복잡도 분석
        
        - **네트워크 복잡도 평가**: Keras를 사용하여 네트워크 훈련 매개변수의 총 수를 추정했습니다.
        - **1D CNN**: 제안된 1D CNN의 훈련 매개변수는 약 300만 개입니다.
        - **CRNN과의 비교**: CRNN 아키텍처는 2D CNN과 LSTM으로 구성되며, 훈련 매개변수가 약 1천만 개로, 제안된 1D CNN보다 3배 많습니다.
        - **CRNN의 복잡도**: 2D 컨볼루션과 LSTM이 CRNN의 네트워크 복잡도를 지배합니다.
        
        ## 3.5.4. 다양한 방법들과의 비교
        
        - **성능 비교**: 제안된 방법을 기존의 딥러닝 기반 방법들과 비교했습니다.
        - **검출 정확도**: 제안된 방법이 모든 클래스에 대해 더 나은 예측 정확도를 달성했습니다.
        - **F1 점수**: 제안된 네트워크가 기존 네트워크보다 높은 평균 F1 점수를 달성했습니다.
        - **네트워크 복잡도**: 제안된 네트워크는 CRNN보다 훨씬 낮은 네트워크 복잡도를 가집니다.
        
        ## 4. 결론
        
        - **1D CNN 개발**: 시계열 ECG 데이터에서 심방세동(AF)을 탐지하기 위한 end-to-end 1D CNN을 개발했습니다.
        - **BN과 풀링 방법의 영향**: BN은 학습을 가속화하지만, 훈련의 불안정성을 초래할 수 있습니다.
        - **풀링 방법**: **평균 풀링은 이전 층의 정보를 유지하고 전달하는 데 유리하며, 맥스풀링보다 안정적**입니다.
        - **성능 평가**: 제안된 1D CNN 방법은 기존 방법들보다 더 나은 교차 검증 검출 정확도를 달성했습니다.
        - **향후 연구 방향**: 고정된 입력 크기가 필요 없는 공간 피라미드 풀링 적용 및 극단적으로 불균형한 데이터셋 문제 해결 방법 탐색.
        
        ### 리뷰
        
        이 논문은 1D CNN을 사용하여 ECG 데이터를 분석하고 심방세동을 탐지하는 데 있어 중요한 기여를 합니다. 네트워크 복잡도 분석은 제안된 모델이 CRNN과 비교하여 훨씬 낮은 계산 비용을 가짐을 보여줍니다. 또한, BN과 풀링 방법에 대한 실험은 모델의 성능과 안정성에 미치는 영향을 잘 보여줍니다. **특히, 평균 풀링이 맥스풀링보다 정보를 더 잘 보존하고 안정적인 훈련을 제공한다는** 점은 주목할 만합니다.
        
        결론적으로, 이 논문은 심방세동 탐지를 위한 효과적인 1D CNN 아키텍처를 제시하며, 이는 ECG 데이터 분석에 있어 중요한 진전을 나타냅니다. 
        
        제안된 모델은 높은 정확도와 낮은 복잡도를 가지며, 이는 실제 의료 환경에서의 적용 가능성을 높입니다. 또한, 향후 연구 방향은 이 분야의 연구를 더욱 발전시킬 수 있는 기회를 제공합니다.
        
    - 번역_**Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL**
        
        블로그에 정리해둠
        
        [https://ashton0410.tistory.com/entry/Deep-Learning-for-ECG-Analysis-Benchmarksand-Insights-from-PTB-XL](https://ashton0410.tistory.com/entry/Deep-Learning-for-ECG-Analysis-Benchmarksand-Insights-from-PTB-XL)
        
    - 리뷰_1 **Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL**
        
        
        **ECG 해석을 위한 딥러닝: PTB-XL의 벤치마크와 통찰**
        
        [https://ar5iv.labs.arxiv.org/html/2004.13701](https://ar5iv.labs.arxiv.org/html/2004.13701)
        
        1. ptb 데이터 셋 
        
         18885명의 환자로 부터 얻은 10초 길이의 임상 12-lead ecg 기록 21837 로  구성되어 있으며  남성이 502% 여성이 48% 
        
        1. 방법론 
        
        2-1 시계열 분류 알고리즘 
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2032.png)
        
        2-2 다중 레이블 분류 메트릭
        
        다중 레이블 분류 메트릭은 크게 샘플 중심 메트릭과 레이블 중심 메트릭으로 구분 
        
        → 주요 차이는 라벨별로 점수를 집계한 다음 샘플별로 집계할지 아니면 반대의 경우인지 
        
        - ecg 분류 알고리즘의 전반적인 성능들
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2033.png)
        
        **CNN(Convolution Neural Network)을 이용한 심전도 데이터 기반 건강상태 모니터링 기술**
        
        [0424 (1).pdf](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/0424_(1).pdf)
        
        해당 논문에서 사용한 방법론 
        
        Convolution Neural Network(CNN)를 사용함
        
        - Convolution Neural Network 구조 특징맵 추출을 위해 Convolutional Layer, Batch-Normalization(BN) Layer, ReLu Activation Layer, Max Pooling Layer를 총 4번 반복하여 구성되었으며. 15개 유형의 분류를 위해 3개 층의 Dense Layer로 구성하 였다. 표 1은 상세 구조를 보여준다
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2034.png)
        
        - 학습데이터와 검증데이터의 비율 8:2
        
        **딥러닝기반 심도 분류의 국내외 동향분석**
        
        [52811306_pv (1).pdf](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/52811306_pv_(1).pdf)
        
        1. 국내동향
        
        인트라- 대상의 심전도 분류 
        
        - 단일 대상 내에서 심전도 신호를 분류하는 것
            - 심장신호의 이상리듬을 의미하는 부정맥 신호를 판별
            - 건국대학교에서는 심장의 건강상태를 평가하는 심전도 신호에서 부정맥을 검출하고자 딥러닝 모델을 제시 및 최적의 ㅣㅂ러닝 파라미터를 구한바 있음
            - 서울대학교에서는 LSTM모델을 이용하여 별토의 특징 추출과정없이 부정맥 여부를 판별하였음
        
        인터- 대상의 심전도 분류 
        
        - 여러 대상간의 심전도 신호를 분류하는것
            - 주로 개인 식별 기술과 같이 연구되고 있음
            - 조선대학교에서는 심전도 신호를 이용한 오토 인코더의 개인 식별 성능에 대한 비교분석을 하였음
            - 국내에서는 딥러닝을 이용한 인터- 대상 심전도 분류 출판 논문이 적은 실정
        
        1. 국외 동향
        - 중국 과학원 대학교는 12-리드 심전도를 9가지 범주로 분류하는 알고리즘을 제안
            - 먼저 전문가로부터 일반 특징과 특정특징을 추출
            - 17층의 CNN모델을 이용하여 특징을 추출한뒤 전문가 특징과 결합한후 XGBoost 분류기에 하습시켜서 예측 결과를 융합함
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2035.png)
        
        - 홍콩 시립대학교는 심박률에 일치시켜 심전도를 분리하고 2개의 연속 심전도 쌍을 dbc 방법으로 2차원 변환하였음
            - 이 2차원 변환 영상을 기반으로 딥러닝 모델을 학습
        
        **딥러닝 네트워크 기반의 오토인코더 전이학습을 이용한 심방세동 검출 모델 개발**
        
        HSIEH 외(2020)는 심방세동을 포함한 부정맥을 분류하는 것을 목표로 하는 open-access 데이터셋인 PhysioNet Challenge 2017 데이터셋[7] 을 이용하여 연구를 수행하였다. 저자들은 Machine Vision, 자연어 처리 등 많은 분야에서 최고성능을 보인 Convolutional Neural Network (CNN)을 적용하여 심방세동을 검출했다. 원래의 데이터 형태인 시계열 데 이터 심전도가 그대로 사용되었고, 10개의 CNN layer와 1개의 Batch Normalization layer, 그리고 3개의 Fully Connected layer를 이용해 총 14 layer로 딥러닝 모델을 구성해 0.78의 F1 score를 얻었다[8]. M Zihlmann 외(2017) 또한 PhysioNet Challenge 2017 데이터셋을 사 용하여 심방세동을 포함한 부정맥을 분류하는 연구를 수행하였다. 먼저 심 전도 데이터는 스펙트로그램(Spectrogram) 기법을 이용하여, 시간과 주파 수 정보를 가진 도메인의 데이터로 변환되었다. 저자들은 CNN에 시간 정 보를 분석하는 분야에서 최고성능을 보였던 Recurrent Neural Network (RNN)를 응용하여 24 layer를 가지는 Convolutional Recurrent Neural Network (CRNN) 모델을 구성하여 0.82의 F1 score를 얻었다[9].
        
        → 이러한 연구는 같은 장치를 이용하여 기록된 내부 데이터 셋에서만 학습하고 성능을 테스트했다는 한계가 존재함 
        
        ⇒ 외부 데이터 셋을 사용하는 경우 성능이 큰폭으로 떨어지는 결과를 보인다. 
        
        - 본문의 저자는 이러한 문제를 완화하기 이해 오토 인코더를 사용함
            
            
            - 오토인코더 기법은 입력데이터의 차원을 합축된 형태로 축소하는 인코더와 축서된 숨교진 표현을 다시 입력과 같은 형태로 복원하는 디코더로 이루어짐
        
        - 이를 다시 전이학습( 미리 학습된 모델에 가중치를 사용하여 더 빠른 학습과 성능을 얻을수 있게하는 학습방법)함
        
        모델 개발의 워크 플로우 
        
        학습된 오토인코더의 모델은 multi-kernel size CNN을 이용한 Fully convolutional network의 구조를 기반으로 ResNet을 응용하여 만 들어졌다. multi-kernel size CNN의 각 블록은 입력을 4개 혹은 5개의 kernel_size를 이용해 비선형화한 후 병합한다. 첫 multi-kernel size CNN 블록에서는 8, 16, 32, 64, 128의 길이를 가지는 kernel size가 사 용되며 이때 각각 16개의 feature map을 생성하며, Concatenate되고 Batch-Normalization layer를 거친다. 두 번째 블록부터는 stride를 2로 설정해 입력을 비선형화 함과 동시에 길이를 1/2로 줄여 표현을 압축한 다. Skip connection은 입력을 각 블록의 입력에 맞춘 크기로 Average pooling하여 진행되며, linear convolution layer를 거치면서 80개의 channel 수 만큼 특징을 추출하고, Batch-Normalization layer를 거친 다. 이후, multi-kernel size CNN과 skip connection을 더하여 다음 층 으로 연결된다. 이 과정이 각 층에서 반복되어 고차원의 압축된 표현을 생 성하며, 다시 이것을 Transpose multi-kernel size CNN을 사용하여 입 력된 신호의 길이를 증가시켜 원래의 신호로 복원 한다. Transpose multi-kernel size CNN블록도 multi-kernel size CNN 블록과 같은 방 법으로 이전 블록의 출력과 skip connection을 더해 다음 블록의 출력으 로 연결한다. Multi-kernel size CNN과, transposed multi-kernel size CNN는 비선형성의 증가를 위해 activation function으로 ELU가 사용되 었고, Skip connection은 입력형태의 원형을 보존하기 위해 linear function이 사용되었다. Batch size는 128이며, learning rate은 0.001로 학습되었다. Loss function은 평균 제곱 오차(Mean squared error)가 사용되었다 (식 2). 오토인코더 모델의 구조는 그림 10에 표시되었고 각 블록에서 사용된 channel 수, kernel size, stride, activation function 과 같은 hyper-parameter 값은 표3에 표시되었다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2036.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2037.png)
        
        **A Study of ECG Classification Algorithm Using Deep Learning**
        
        **딥러닝을 사용한 ECG데이터 분류 알고리즘**
        
        [jkits-2022-17-1-71 (1).pdf](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/jkits-2022-17-1-71_(1).pdf)
        
        합성곱신경망(cnn)을 사용하였다. 
        
        ****ECG data dependency for atrial fibrillation detection based on residual networks****
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2038.png)
        
    - 리뷰2_**Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL**
        
        **Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL _20221227**
        
        **ECG 해석을 위한 딥러닝: PTB-XL의 벤치마크와 통찰**
        
        [https://ar5iv.labs.arxiv.org/html/2004.13701](https://ar5iv.labs.arxiv.org/html/2004.13701)
        
        1. **배경**
            - 자동 ECG 해석의 발전이 대규모 데이터셋 부재와 정의된 평가 절차의 부족으로 제한됨.
            - PTB-XL 데이터셋은 이 문제를 해결하기 위해 제공됨.
        2. **방법론**
            - 데이터셋: PBT-XL. 대규모 공개 ECG 데이터셋 사용.
            - 다양한 ECG 해석 작업에 대한 벤치마킹 수행.
                - ECG을 분석해 심장 상태 예측
                - 연령 및 성별 예측
                - 신호 품질 평가
            - CNN, 특히 **ResNet과 Inception** 기반 아키텍처의 성능 평가.
        3. **결과**
            - **CNN**이 기능 기반 알고리즘보다 뛰어난 성능을 보임.
            - 벤치마킹을 통해 숨겨진 계층화, 모델 불확실성, 해석 가능성에 대한 통찰력 제공.
            - **ICBEB2018 데이터셋** 에 대한 벤치마킹을 통해 PTB-XL 데이터셋이 전이 학습에 유용함을 보여준다.
        4. **결론 및 미래의 방향성**
            - 정량적 정확도뿐만 아니라 해석 가능성과 모델의 강건성도 중요.
            - PTB-XL 데이터셋을 ECG 분석 알고리즘 벤치마킹을 위한 표준 자원으로 설정하는 것을 목표로 함.
            - 이를 통해 의료 분야에서 딥러닝 알고리즘의 임상적 영향을 높일 수 있을 것으로 기대됨
    - ECG 관련 논문 리스트업
        
        
        1. ECG에서 수동으로 AF를 스크리닝하는 것은 시간이 많이 소요되고 오류가 발생하기 쉽습니다. 이러한 한계를 극복하기 위해, 인공 지능 기술을 사용하는 컴퓨터 지원 진단 시스템이 개발되었습니다.
            
            [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8583162/#:~:text=Atrial fibrillation ,for automated detection of AF](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8583162/#:~:text=Atrial%20fibrillation%20,for%20automated%20detection%20of%20AF)
            
        
        1. **네이처**에 발표된 연구는 심방세동 탐지를 위한 딥러닝 기반 알고리즘의 개발에 대해 설명합니다. 이 모델들은 종종 제한된 데이터셋으로 훈련되었고 특정 조건 내에서 평가되었다고 언급하고 있는데, 이는 임상 보고서를 포함한 종합적인 데이터셋이 필요할 수 있다는 것을 시사합니다
            
            [https://www.nature.com/articles/s41598-021-97308-1#:~:text=To detect AF%2C deep learning,and were evaluated within the](https://www.nature.com/articles/s41598-021-97308-1#:~:text=To%20detect%20AF%2C%20deep%20learning,and%20were%20evaluated%20within%20the)
            
        
        1. **ECG-AI** 컨볼루션 신경망은 매사추세츠 종합병원에서 종단적 일차 진료를 받는 환자들의 12-리드 ECG를 사용하여 5년 간의 심방세동 위험을 추론하기 위해 컨볼루션 신경망(ECG-AI)을 훈련했습니다. 이는 임상 환경에서 딥러닝의 실용적인 적용을 보여줍니다
            
            [https://pubmed.ncbi.nlm.nih.gov/34743566/#:~:text=Methods%3A We trained a convolutional,MGH](https://pubmed.ncbi.nlm.nih.gov/34743566/#:~:text=Methods%3A%20We%20trained%20a%20convolutional,MGH)
            
        
        1. Acharya 등은 ECG 신호 분류를 위한 새로운 딥 CNN을 설계했고, 다른 연구에서는 심장 박동을 네 클래스로 분류하기 위해 딥 빌리프 네트를 사용했습니다. 실시간으로 심방세동을 탐지하기 위해 사용된 새로운 딥러닝 접근법이 제시되었습니다.
            
            [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7313287/#:~:text=Acharya et al,In this](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7313287/#:~:text=Acharya%20et%20al,In%20this)
            
        2. 2019년에는 다학제 연구자 그룹이 정상 리듬에서 기록된 단일 ECG를 기반으로 심방세동 환자를 식별할 수 있는 딥러닝 알고리즘을 개발했습니다. 심방세동 조기 탐지의 가능성.
            
            [https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)01802-5/fulltext#:~:text=In 2019%2C an interdisciplinary group,ECG recorded in sinus rhythm](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)01802-5/fulltext#:~:text=In%202019%2C%20an%20interdisciplinary%20group,ECG%20recorded%20in%20sinus%20rhythm)
            
    - 논문들의 아키텍처 및 레이어
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2039.png)
        
        [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180882/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180882/)
        
        | Layers | Parameters | Activation |
        | --- | --- | --- |
        | Conv1D | Filter 32/kernel 5 | ReLU |
        | BN |  |  |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 32/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 64/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 64/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 128/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 128/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Dropout | 0.5 |  |
        | Conv1D | Filter 256/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Conv1D | Filter 256/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Dropout | 0.5 |  |
        | Conv1D | Filter 512/kernel 5 | ReLU |
        | Maxpooling | 2 |  |
        | Dropout | 0.5 |  |
        | Conv1D | Filter 512/kernel 5 | ReLU |
        | Flatten |  |  |
        | Dense | 128 | ReLU |
        | Dropout | 0.5 |  |
        | Dense | 32 | ReLU |
        | Dense | 4 | Softmax |
    
- kaggle
    - kaggle 1
        
        # 20231112 장보경 코드
        
        ## 사용한 코드
        
        - 기존 캐글 노트북 코드 [https://www.kaggle.com/code/khyeh0719/ptb-xl-dataset-wrangling](https://www.kaggle.com/code/khyeh0719/ptb-xl-dataset-wrangling)
        - 멘토님께서 주신 PTB 가이드라인 코드파일
        
        ## 코드 내용
        
        - 캐글 코드에 한국어 주석 처리 + 설명 + 시각화 아웃풋 캡쳐본 추가
        - 가이드라인의 내용과 노트북 내용 비교
        
        ---
        
        접근이 많이 다르긴 하지만 기본 EDA 방식에 있어서는 많이 참고 하면 좋을 것 같다.
        
        → 코드는 슬랙에 올렸음.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2040.png)
        
        - 깃헙에 접근이 안되서 나중에 업로드 하는 걸로.
        
    - kaggle 2 (1D CNN)
        
        [https://www.kaggle.com/code/jraska1/ptb-xl-ecg-1d-convolution-neural-network](https://www.kaggle.com/code/jraska1/ptb-xl-ecg-1d-convolution-neural-network)
        
        이번엔 캐글 노트북에 바로 수정하고 돌린 후에 공유하려고 했는데, 캐글 자체 이슈로 돌아간 후에 저장이 안되서 일단 코드만 공유했다.
        
    - kaggle pytorch CNN 리스트업
        
        [https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism](https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism)
        
        [https://www.kaggle.com/code/khyeh0719/auto-ecg-diagnose-with-deep-learning](https://www.kaggle.com/code/khyeh0719/auto-ecg-diagnose-with-deep-learning)
        
    - kaggle 코드 분석
        
        ****심방세동 감지(99% Acc)****
        
        [https://www.kaggle.com/code/thinhdvo/atrial-fibrillation-detection-99-acc](https://www.kaggle.com/code/thinhdvo/atrial-fibrillation-detection-99-acc)
        
        - 필요 없다고 판단되는 컬럼을 드랍하였다.
            - 먼저 df.copy()를 통해 전처리용 df를 따로 생성
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2041.png)
        
        - 나이 데이터를 전부 구간대 데이터로 변환
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2042.png)
        
        - 키와 몸무게 역시 구간 데이터로 변환 하였다.
            - 그런데 키와 몸무게가 심방세동과 상관관계가 있을까?
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2043.png)
        
        - 정상 ecg와 비정상(심방세동 ecg)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2044.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2045.png)
        
        - 모델링
            - 머신러닝 알고리즘 중에 `RandomForestClassifier()` 를 사용하였다.
            - `GridSearchCV` 를 이용하여 파라미터를 조절하였다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2046.png)
        
        - 혹시나 해서 일단 캐글의 코드 그대로 돌려 보았다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2047.png)
        
    
- 기타
    - 관련 기사 ****PTB-XL, a large publicly available electrocardiography dataset****
        
        [https://www.nature.com/articles/s41597-020-0495-6](https://www.nature.com/articles/s41597-020-0495-6)
        
    - PYTORCH의 TRANSFORMER를 사용한 심방세동
        
        
        [https://towardsdatascience.com/heart-disease-classification-using-transformers-in-pytorch-8dbd277e079](https://towardsdatascience.com/heart-disease-classification-using-transformers-in-pytorch-8dbd277e079)
        
        아티클 사이트 
        
        [https://github.com/bh1995/AF-classification](https://github.com/bh1995/AF-classification)
        
        해당 아티클에서 공유한 github 페이지 
        
    - 기존 부정맥 분류 연구들
        
        
        | 연구자 | 출처 | 날짜 | 연구 방법 | 데이터셋 | 주요 결과 및 특징 |
        | --- | --- | --- | --- | --- | --- |
        | S. S. Aphale, E. John, T. Banerjee | IEEE International Midwest Symposium on Circuits and Systems (MWSCAS), 2021 | 2021 | ArrhyNet 사용, SMOTE 기법 | 미상 | 92.73% 정확도 |
        | E. Çimen | Eskisehir Technical University Journal of Science and Technology A-Applied Sciences and Engineering, Vol. 22, No. 1 | 미상 | 2-D CNN과 VGG16을 이용한 전이 학습, v-SVM 분류기 | 미상 | 정상 및 부정맥 신호 각각 87%, 93% 정확도, 평균 90.42% |
        | A. Isin, S. Ozdalili | Procedia Computer Science, Vol. 120 | 2017 | AlexNet을 이용한 전이 학습 | 미상 | 92% 정확도 |
        | A. Mustaqeem, S. M. Anwar, M. Majid | Computational and Mathematical Methods in Medicine, 2018 | 2018 | 랜덤 포레스트 기반 래퍼 방법, SVM 기법 | UCI 데이터베이스 | 92.07% 정확도 (90:10 데이터 분할) |
        | A. Y. Hannun et al. | Nature Medicine, Vol. 25, No. 1 | 2019 | 34-층 DNN 모델 | 미상 | 12 리듬 클래스 분류, AUC 0.97 |
        | S. Ilbeigipour, A. Albadvi, E. Akhondzadeh Noughabi | Journal of Healthcare Engineering, 2021 | 2021 | MIT-BIH 데이터베이스 사용, 의사결정 트리, 랜덤 포레스트, 로지스틱 회귀 | MIT-BIH 데이터베이스 | 평균 88.7% 정확도, 92.5% 정밀도 |
        | P. Lu et al. | Journal of Healthcare Engineering, 2021 | 2021 | KecNet 모델, CNN 구조 | 미상 | 99.31% 정확도 |
        | P. Kanani, M. Padole | Procedia Computer Science, Vol. 171 | 2020 | 1-D CNN 모델, 시계열 변환 | 미상 | 99% 정확도, 과적합 없음 |
        
        이 표는 각 연구의 주요 내용을 간략하게 요약한 것으로, 더 자세한 정보는 각 연구의 원문을 참조하시는 것이 좋습니다. 연구 방법, 데이터셋, 정확도 등의 세부 사항은 연구마다 다양하며, 각 연구의 방법론과 결과를 비교 분석함으로써 심장 부정맥 탐지를 위한 다양한 접근 방식을 이해할 수 있습니다.
        
    - transformer 코드 아티클 정리
        
        # **PyTorch의 Transformer를 사용한 심장병 분류**
        
        이 글에서는 심전도(ECG/EKG) 데이터를 이용한 심방세동(AF) 분류 시스템을 구현하고 논의합니다. 분류 시스템은 이진형(정상동리듬, AF)이며 PyTorch 프레임워크를 사용하는 변환기 네트워크를 기반으로 합니다. 이 기사의 구조는 다음과 같습니다. 먼저 당면한 문제와 사용된 데이터를 소개하고, 두 번째로 네트워크의 기술적 세부 사항을 제시하고, 세 번째로 네트워크 구현 및 교육을 제시합니다. , 마지막으로 결과 발표와 토론이 이어집니다. 이 프로젝트의 저장소는 [https://github.com/bh1995/AF-classification](https://github.com/bh1995/AF-classification) 에서 찾을 수 있습니다 .
        
        # **심방세동 분류 문제**
        
        심장병은 전 세계적으로 사망의 주요 원인 중 하나입니다[1]. 심장 질환은 종종 부정맥이라고 불리는 심장 행동의 불규칙성을 유발합니다. 모든 유형의 부정맥 중에서 심방세동(AF)은 가장 흔한 심장 부정맥이며 인구의 2~4% 범위에서 유병률이 있는 것으로 추정됩니다. 향후 인구의 평균연령이 높아질수록 유병률은 선형적으로 증가할 것으로 예상된다. 현재 통계를 고려할 때 유럽계 개인이 일생 동안 심방세동(AF)으로 고통받을 확률은 대략 37%입니다[2]. 표준 심장 모니터링 기술은 심전도(ECG/EKG)이며, 이는 심장의 전기 신호를 모니터링하므로 임상의가 부정맥을 진단하는 데 매우 유용할 수 있습니다. 이 프로젝트의 목적은 임상의가 라벨을 붙인 실제 공개 ECG 데이터를 활용하여 딥 러닝 모델을 훈련시키는 것입니다.
        
        이 프로젝트에 사용된 데이터는 중국 생리학적 신호 챌린지 2018(CPSC2018)[3], 상트페테르부르크 INCART 12-리드 부정맥 데이터베이스[4], 조지아 12-리드 ECG 챌린지 데이터베이스(CinC2020)[5 등 여러 소스에서 가져온 것입니다. ], PhysioNet Computing in Cardiology Challenge 2017(CinC2017) 데이터세트[5], 상황별 부정맥 데이터베이스(CACHET-CADB)[6].
        
        사용된 데이터 소스는 다양한 샘플링 속도와 다양한 장치를 사용하여 ECG 데이터를 수집했기 때문에(일부는 홀터 모니터를 사용하고 다른 일부는 병원 등급 12 리드 ECG를 사용함) 이로 인해 각 데이터 소스의 특성이 약간 다릅니다. 딥러닝 모델의 학습 능력을 높이려면 모든 데이터를 정규화 방식으로 처리하는 것이 현명합니다. 본 프로젝트에서는 [0.5, 40] 범위의 주파수만 유지되도록 모든 신호에 대역 통과 필터(FIR 필터)를 사용했습니다. 또한 신호는 300Hz의 속도로 리샘플링되고 길이가 10초인 개별 세그먼트로 분리됩니다(따라서 모델에 대한 각 입력은 길이가 3000개 데이터 포인트인 1차원 배열입니다). 그런 다음 각 10초 세그먼트는 모든 값이 0과 1 사이에 있도록 정규화됩니다. 아래 그림은 2개의 원래(원시) 신호(AF 1개와 일반 1개)와 사전 처리 후의 동일한 2개 신호를 보여줍니다.
        
        ![https://miro.medium.com/v2/resize:fit:700/1*bkpJ5BBu6-0e71955Teudw.jpeg](https://miro.medium.com/v2/resize:fit:700/1*bkpJ5BBu6-0e71955Teudw.jpeg)
        
        AF의 원시 신호(왼쪽)와 건강한 정상 동박동(오른쪽)을 보여주는 플롯.
        
        ![https://miro.medium.com/v2/resize:fit:700/1*wdm9gs4UeCTupVvHoorZMA.jpeg](https://miro.medium.com/v2/resize:fit:700/1*wdm9gs4UeCTupVvHoorZMA.jpeg)
        
        AF의 사전 처리된 신호(왼쪽)와 건강한 정상 동박동(오른쪽)을 보여주는 플롯.
        
        첫 번째 이미지의 신호 플롯은 표준적이고 명확한 10초 길이의 ECG 기록을 보여줍니다. 플롯의 y축은 전압 차이를 나타내고 x축 샘플링 속도(이 경우 300Hz)를 나타냅니다. 플롯에서 볼 수 있듯이 특정 사람의 심장 전압은 다른 사람에 비해 다릅니다. 우리 인간의 경우 y축의 진폭을 보는 대신 x축의 패턴을 찾는 것이 분명합니다. 그러나 딥 러닝 모델은 편견 없는 방식으로 특징을 계산하므로 제공된 모든 정보를 순진하게 사용합니다(진폭이 일반적으로 부정맥 분류와 관련이 없더라도). 이것이 데이터를 주어진 간격(예: [-1,1] 또는 [0,1])으로 정규화하는 것이 매우 중요한 이유입니다. 위 그림은 AF의 예를 보여줍니다. 이는 일부 심장 박동 사이의 불규칙한 간격에서 볼 수 있습니다. 이 경우는 아마추어라도 AF 신호의 이상을 명확하게 볼 수 있는 '간단한' 예입니다.
        
        이 프로젝트에 사용된 전체 데이터 세트는 **[링크를](https://drive.google.com/drive/folders/1PONXv-dtV26nqTFz_EBEhEufLQiETycF?usp=sharing)** 통해 다운로드할 수 있습니다 . 데이터는 모두 전처리되어 **.h5** 파일 형식의 배열로 저장됩니다. 아래 코드 조각은 데이터 세트를 로드하는 방법의 예를 보여줍니다.
        
        이 프로젝트의 데이터세트를 로드하는 중입니다.
        
        # **모델 설명**
        
        이 프로젝트에서는 두 가지 모델이 테스트되었습니다. 둘 다 동일한 기본 아키텍처를 따르지만 서로 다른 입력을 사용합니다. 첫 번째 모델은 사전 처리된 ECG 신호를 유일한 입력으로 사용하는 반면, 두 번째 모델은 사전 처리된 ECG 신호와 해당 신호에 대한 소위 RRI(RR 간격)를 사용합니다. RRI는 심장 박동과 연속 심장 박동 사이의 예상 시간(초)입니다. RRI는 Pan-Tompkins 알고리즘[10]을 사용하여 신호에서 R-peak(각 심장박동의 중간 위치)를 추정한 후 다음 방정식을 사용하여 계산됩니다.
        
        ![https://miro.medium.com/v2/resize:fit:143/1*7wPwQ-O0n-TSdeiLS5w7pg.png](https://miro.medium.com/v2/resize:fit:143/1*7wPwQ-O0n-TSdeiLS5w7pg.png)
        
        여기서 *Rn* 은 주어진 피크이고 *fs* 는 주파수(샘플링 속도)입니다. 입력 RRI 시퀀스의 설정된 길이를 모델에 유지하기 위해 전체 RRI 중 처음 10개가 사용되고 10개 미만이 발견되면 0으로 채워집니다.
        
        프로젝트에 사용된 모델은 [8]의 작업에서 영감을 받았습니다. 이 모델은 1) 일련의 컨벌루션 레이어로 구성된 임베딩 네트워크, 2) 변환기 인코더 레이어의 스택, 3) 일련의 완전히 연결된 레이어로 구성된 분류 헤드 모듈로 구성됩니다. 모델 아키텍처 및 모델 교육을 위한 코드는 [여기에서](https://github.com/bh1995/AF-classification) 찾을 수 있습니다 .
        
        **임베딩** : 임베딩 네트워크는 원래 ECG 신호에 1차원 컨볼루션을 적용하여 일련의 임베디드 표현( *x[0], …, x[n]* )을 생성합니다. *이러한 내장된 표현은 위치 인코딩( p[0], …, p[n]* ) 과 함께 합산되어 원본 변환기 문서[7]에서 수행된 것처럼 각 시퀀스의 순서를 나타냅니다. 결과는 위치 인코딩 임베딩 *e =* ( *x[0]+p[0], …, x[n]+p[n]* )이며, 이는 적층 변환기 인코더 레이어 모듈에 대한 입력으로 사용됩니다.
        
        **변압기 인코더 레이어:** 변압기 모듈은 변압기 인코더 레이어의 스택으로 구성됩니다. 여기서 각 인코더는 다중 헤드 셀프 어텐션 메커니즘 하위 레이어로 구성됩니다. 시행 착오 후에 4개의 인코더 레이어와 헤드가 좋은 결과를 생성하는 것으로 나타났습니다. . 다시 말하지만, 트레일 앤 에러 후에 인코더 내의 임베딩 차원은 *64* 로 선택되었고 인코더 레이어 내의 피드포워드 네트워크의 차원은 *512* 로 선택되었습니다 . 변환기 모듈의 출력은 [시퀀스 길이, 배치 크기, 임베딩] 모양의 주의 가중치 텐서입니다. 출력은 분류 헤드 모듈에 공급되도록 어떤 방식으로든 변환되어야 합니다. 즉, [배치 크기, 임베딩] 모양으로 변환되어야 합니다. 두 가지 방법이 시도되었습니다. 첫 번째는 단순히 시퀀스 차원에서 텐서의 평균을 취하는 것이고 두 번째는 [9]의 작업에서 제안한 대로 자기 주의 풀링 레이어를 사용하는 것입니다. 두 방법 모두 좋은 성능을 보였지만 self attention pooling 레이어를 활용하는 것이 가장 좋은 결과를 얻었습니다. 자기 주의 풀링 계층은 인코더 시퀀스의 특징에 대해 학습된 평균인 임베딩을 생성하는 변환기 모듈의 출력에 적용됩니다.
        
        **분류 헤드:** 자기 주의 풀링의 출력은 예측에 사용되는 로짓을 생성하기 위해 최종 분류 헤드에 대한 입력으로 사용됩니다. 최종 분류 헤드는 사이에 단일 드롭아웃 레이어가 있는 두 개의 완전히 연결된 레이어입니다.
        
        # **훈련**
        
        모델은 손실 함수로 이진 교차 엔트로피를 사용하고 최적화 도구로 AdamW 알고리즘을 사용하여 훈련되었습니다( *β1=0.9, β2=0.98, ε=10^(-9)* ). 초기 학습률은 10^(-3)이 사용되었으며 학습 스케줄러를 사용하여 처음 30 에포크 이후부터 5 에포크마다 학습 속도를 5%씩 감소시켰습니다. 데이터 세트에는 총 52,370개의 10초 길이의 ECG 신호가 있으며, 여기서 정상과 AF 레이블의 비율은 대략 균일합니다. 데이터세트의 85%(44,514)는 훈련용으로 무작위로 선택되었고 나머지 15%(7,855)는 테스트용으로 사용되었습니다. 배치 크기는 10이 사용되었으며 클라우드 컴퓨팅 환경에서 Tesla P100 GPU를 사용하여 훈련이 수행되었습니다.
        
        # **결과**
        
        두 모델에 대한 결과는 아래에 제시되어 있습니다. 먼저 원시 ECG 신호만 사용하는 모델과 원시 ECG 신호 및 해당 신호에 대한 RRI를 사용하는 모델이 있습니다.
        
        **원시 ECG 신호 사용:**
        
        아래 플롯에서 볼 수 있듯이 모델은 상당한 과적합 없이 여러 시대에 걸쳐 일관된 학습을 보여주었습니다. 테스트 정확도도 지속적으로 증가하는 것으로 나타났습니다. 이 작업은 질병의 이진 분류이기 때문에 민감도와 특이도가 성능에 대한 더 나은 척도임은 틀림없습니다. 훈련 종료 시 테스트 데이터의 민감도 와 **특이도 는 각각 85.6** 과 **92.7** 이었습니다 . 학습 시간은 에포크당 약 **178초 였습니다.**
        
        ![https://miro.medium.com/v2/resize:fit:700/1*RSY1flkcNs3EAZWZauDY4w.png](https://miro.medium.com/v2/resize:fit:700/1*RSY1flkcNs3EAZWZauDY4w.png)
        
        시대별 학습(주황색) 및 테스트(파란색) 데이터에 대한 손실 곡선.
        
        ![https://miro.medium.com/v2/resize:fit:700/1*s9Kui6GvpBbWlPrG-Nvi8w.png](https://miro.medium.com/v2/resize:fit:700/1*s9Kui6GvpBbWlPrG-Nvi8w.png)
        
        여러 시대에 걸쳐 테스트 데이터에 대한 모델의 정확도.
        
        **원시 ECG 신호 및 RRI 사용:**
        
        훈련/추론 중에 RRI 기능을 사용하면 모델 성능이 향상되었습니다. 아래 플롯은 모델이 과적합의 명확한 징후를 보여주지만 이 과적합은 에포크에 대한 테스트 데이터의 모델 정확도에 부정적인 영향을 미치지 않습니다. 훈련 종료 시 테스트 데이터의 민감도 와 **특이도는 각각 96.9** 와 **95.6** 이었습니다 . 학습 시간은 에포크당 약 **1550초 였습니다.**
        
        ![https://miro.medium.com/v2/resize:fit:700/1*zulZQCEZMeA4wIAajPBREA.png](https://miro.medium.com/v2/resize:fit:700/1*zulZQCEZMeA4wIAajPBREA.png)
        
        시대별 학습(주황색) 및 테스트(파란색) 데이터에 대한 손실 곡선.
        
        ![https://miro.medium.com/v2/resize:fit:700/1*9_wlpfjO1rxJd0PozGlcQQ.png](https://miro.medium.com/v2/resize:fit:700/1*9_wlpfjO1rxJd0PozGlcQQ.png)
        
        여러 시대에 걸쳐 테스트 데이터에 대한 모델의 정확도.
        
        # **결론**
        
        결과를 보면 변환기 인코더 기반 네트워크가 10초 길이의 ECG 신호에서 AF를 분류하는 데 잘 작동한다는 것이 분명해졌습니다. RRI 기능을 사용하면 훈련 및 추론 시간이 거의 9배 증가하는 대신 모델 성능이 크게 향상된다는 것도 분명합니다. 이러한 시간 증가는 R-피크 검출을 수행하는 데 사용되는 Pan-Tompkins 알고리즘이 느리기 때문입니다.
        
    

# ****[2]**** EDA

- 공통
    - 데이터 전처리 및 가공 (가이드)
        
        **guideline-PTB-XL**
        
        [guideline-PTB-XL.ipynb](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/guideline-PTB-XL.ipynb)
        
        데이터 수집 단계 
        
        1. wfdb 패키지 설치
        
         
        
        - PTB-XL의 전처리를 위하여 필요한 패키지
        - [https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb](https://github.com/MIT-LCP/wfdb-python/blob/main/demo.ipynb)
        
        1. 패키지 import 
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2048.png)
        
        1. metadata 읽기
        
        - 다운로드 받은 **PTB-XL 데이터에는 ptbxl_database.csv 파일이 있습니다.**
        - 해당 csv 파일을 pandas를 이용하여 열면, 맨 마지막 **두개의 컬럼 filename_lr, filename_hr**이 있습니다.
        - 500Hz로 수집된 데이터를 사용할 것이기 때문에 **filename_hr 컬럼의 경로를 이용하여 데이터를 읽으면 됩니다.**
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2049.png)
        
        1. 신호 데이터 읽기 
        - 샘플 하나 읽어오기
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2050.png)
        
        signal 변수에는 신호 데이터가, meta 변수에는 여러가지 신호에 대한 정보 데이터가 담겨져 있음 
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2051.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2052.png)
        
        signal 변수에는 numpy array로 저장이 된  5000x12 형태 
        
        - 데이터 슬라이싱
        
        ecg데이터는 총 12개의 신호를 담고있음 
        
        그중 우리는 맨 첫번째 신호인 lead1 신호를 사용할것
        
        (갤럭시 워치나 애플워치와 같이 두팔에서 수집된 신호)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2053.png)
        
        ⇒ 최종 저당되는 데이터는(N, 5000)개로 모든 데이터를 읽어서 하나의 array로 담아 저장 하면 될것 
        
    - 10초 길이의 ECG 데이터
        - **신호의 시간적 연속성**: RNN이나 LSTM과 같은 순환 신경망 모델은 시간에 따른 데이터의 연속성을 모델링하는 데 적합합니다. 이러한 모델은 짧은 기간의 데이터에서도 패턴을 학습할 수 있습니다.
        - **형태학적 특징**: CNN은 ECG 신호에서의 피크, 파동 및 다른 형태학적 특징들을 잘 포착할 수 있습니다. 짧은 기간의 데이터에서도 중요한 형태학적 정보를 학습할 수 있기 때문에, CNN은 부정맥을 탐지하는 데 유용할 수 있습니다.
        - **전체 시퀀스의 컨텍스트**: Transformer 모델은 전체 시퀀스를 한 번에 처리할 수 있는 능력을 가지고 있습니다. 이 모델은 각 시점에서 전체 ECG 신호의 컨텍스트를 고려할 수 있어, 10초 정도의 짧은 신호에서도 효과적일 수 있습니다.
        
        가장 효과적인 접근 방법은 여러 모델을 실험하고, 그 결과를 비교하여 가장 적합한 모델을 선택하는 것입니다. 실제 의료 현장에서의 사용 가능성, 실시간 분석 능력, 모델의 해석 가능성 등 다른 요소들도 고려해야 합니다.
        
    - 나이 성별 심방세동 연관성
        
        구글 탐색결과
        
        심방세동 가진 환자의 56.6%가 65세 이상이었다. **모든 연령 대에서 남 자가 여자보다 빈도가 높았고, 특히 남녀 모두 80세 이 상에서 빈도가 가장 높았다**.
        
        심방세동(AFib)은 심장의 심방이 심실의 속도에 맞지 않게 뛸 때 발생하는 불규칙적인 심장 박동 유형입니다. CDC에 따르면 **65세 미만 인구 중 약 2%와 만 65세 이상 인구 중 약 9%**가 심방세동을 앓고 있습니다
        
    - EDA(guidline, 보경님 코드, kaggle)
        
        # 성식님의 eda 제안
        
        **eda**
        
        - 기존 논문이나 블로그, 캐글 참조 및 구현
        
        **전처리 및 가공** 
        
        - 활용가능한 데이터에 결측치가 없음
        - 주어진 가이드를 통해서 ptb,sph를 라벨링
        - 기본 데이터 프레임만들기
            - 날짜, 환자id, 성별, 나이, 심박수 데이터, 라벨(심방세동여부)
                
                성별 남성이 0 여성이 1 (근거 데이터 설명에 남, 여 비율)
                
                시계열 데이터 활용 가능하게 변경
                
            - PTB와 SPH의 컬럼, 데이터값 동일하게 맞추기
                
                PTB는 컬럼 소문자, sph는 맨앞 대문자 존재
                
                성별 ptb는 숫자 sph는 문자
                
        - 데이터셋 나누기 및 검증 코드작성
        - 데이터 테스트
            - 간단한 모델로 ptb와 sph의 심박수 데이터가 검증 자료로 작동하는지 체크하기
        
        # 가이드라인 1. ptb-xl
        
        [guideline-PTB-XL.ipynb](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/guideline-PTB-XL%201.ipynb)
        
        # 가이드라인 2. sph
        
        [guideline-SPH.ipynb](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/guideline-SPH.ipynb)
        
        # 보경님 코드(with kaggle)
        
        [ptb-xl-wrangling-annie-kg1.ipynb](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/ptb-xl-wrangling-annie-kg1.ipynb)
        
    - [train.py](http://train.py) 오류 해결
        
        
        Error occurred: train_model() missing 1 required positional argument: 'num_epochs’
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2054.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2055.png)
        
    - [test.py](http://test.py) 오류 해결
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2056.png)
        
        모델 상태 불러오는 코드가 test_model 안에 있어야 함.
        
    - PTB-XL-column 분석과 sph**-column** 분석
        
        **PTB-XL-column**
        
        0   ecg_id                        심전도 데이터의 고유 ID.
        
        1   patient_id                   환자의 고유 ID.
        2   age                             환자의 나이.    
        3   sex                              환자의 성별
        
        4   height                         환자의 키
        5   weight                         환자의 몸무게.
        6   nurse                           간호사 정보 또는 ID.
        7   site                               ECG 측정 장소.
        8   device                          ECG 측정에 사용된 기기 정보
        9   recording_date            ECG 측정 날짜 및 시간
        10  report                         ECG 결과 또는 보고서
        11  scp_codes                   ECG 결과와 관련된 코드 또는 부호
        12  heart_axis                   심장 축 정보 
        13  infarction_stadium1   심근 경색 1단계 정보.
        14  infarction_stadium2   심근 경색 2단계 정보.
        15  validated_by               확인자 또는 승인자 정보
        16  second_opinion          CG 결과에 대한 다른 의견 또는 판단
        
        17  initial_autogenerated_report   초기 자동 생성된 보고서
        
        18  validated_by_human    인간에 의한 확인 또는 승인 여부
        
        19  baseline_drift               기준선 드리프트 정보
        20  static_noise                  정전 잡음 정보
        21  burst_noise                  돌발 잡음 정보
        22  electrodes_problems   전극(전기적 연결) 문제 정보       
        23  extra_beats                 추가 심방 박동 정보
        24  pacemaker                 페시메이커 사용 여부 정보                                                                               (심장의 비정상적인 리듬을 조절하기 위해 사용되는 의료 장치)
        25  strat_fold                   분할(fold) 또는 구분 정보
        
        26  filename_lr                낮은 해상도(저해상도) 버전의 ECG 파일 경로          
        27  filename_hr               높은 해상도(고해상도) 버전의 ECG 파일 경로.
        
        **sph-column**
        
        1. 'ECG_ID': 심전도 (Electrocardiogram)의 고유 식별자
        2. 'AHA_Code': American Heart Association (AHA)에서 정의한 심장 질환 코드. 
        3. 'Patient_ID': 환자의 고유 식별자
        4. 'Age': 환자의 나이
        5. 'Sex': 환자의 성별 
        6. 'N': 
        7. 'Date': 데이터가 기록된 날짜
        
    
- PTB_XL
    - PTB-XL dataset 설명
        
        [https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data)
        
        ### ****Data Acquisition****
        
        1. 원시 신호 데이터는 독점적인 압축 형식으로 기록 및 저장되었습니다. 모든 신호에 대해 오른쪽 팔에 기준 전극이 있는 **표준 리드 12개 세트(I, II, III, AVL, AVR, AVF, V1, ..., V6)를 제공**했습니다.
        2. 해당 일반 메타데이터(예: 나이, 성별, 체중, 키)는 데이터베이스에 수집되었습니다.
        3. 각 기록에는 보고서 문자열(심장 전문의에 의해 생성되거나 심전도 장치에 의한 자동 해석)이 주석으로 달렸으며, 이 문자열은 표준화된 SCP-ECG 문(scp_codes) 집합으로 변환되었습니다. 대부분의 기록에서 심장의 축(heart_axis)과 경색 경기장(infarction_stadium1 및 infarction_stadium2, 존재하는 경우)도 추출되었다.
        4. 기록의 상당 부분은 두 번째 심장 전문의에 의해 검증되었습니다.
        5. 모든 기록은 기술 전문가가 주로 신호 특성에 초점을 맞춰 검증했습니다.
        
        ### ****Data Preprocessing****
        
        1. **심전도와 환자는 고유 식별자(ecg_id 및 patient_id)로 식별**됩니다. 
        2. 메타데이터의 개인 정보(예: 기록의 검증 심장 전문의, 간호사, 기록 장소(병원 등)의 이름)는 가명 처리되었습니다. 
        3. **생년월일은 심전도 기록 당시의 나이로만 표시되며, 89세 이상의 나이는 HIPAA 표준에 따라 300세 범위에서 표시됩니다.** 
        4. 또한 **모든 ECG 기록 날짜는 각 환자마다 무작위 오프셋으로 이동되었습니다.** 
        5. 기록에 주석을 달기 위해 사용된 ECG 문장은 SCP-ECG 표준을 따랐습니다.
        
        ### ****Data Description****
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2057.png)
        
        ptbxl
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100
        ├── 00000
        │   │   ├── 00001_lr.dat
        │   │   ├── 00001_lr.hea
        │   │   ├── ...
        │   │   ├── 00999_lr.dat
        │   │   └── 00999_lr.hea
        │   ├── ...
        │   └── 21000
        │        ├── 21001_lr.dat
        │        ├── 21001_lr.hea
        │        ├── ...
        │        ├── 21837_lr.dat
        │        └── 21837_lr.hea
        └── records500
              ├── 00000
              │     ├── 00001_hr.dat
              │     ├── 00001_hr.hea
              │     ├── ...
              │     ├── 00999_hr.dat
              │     └── 00999_hr.hea
              ├── ...
              └── 21000
                     ├── 21001_hr.dat
                     ├── 21001_hr.hea
                     ├── ...
                     ├── 21837_hr.dat
                     └── 21837_hr.hea
        
        이 데이터 세트는 1만 8885명의 환자에서 얻은 10초 길이의 임상 12-리드 ECG 기록 21837개로 구성되어 있으며, 52%는 남성, 48%는 여성이고 연령은 0~95세(중앙값 62, 사분위수 간 범위 22)입니다. 이 데이터 세트의 가치는 다양한 동시 발생 병리를 포괄적으로 수집한 결과일 뿐만 아니라 건강한 대조군 샘플도 상당수 포함했기 때문입니다. 진단 분포는 다음과 같으며, 간결성을 위해 수퍼클래스로 집계된 진단 진술로 제한했습니다(참고: 레코드당 여러 개의 레이블이 있을 수 있으므로 진술의 합계가 레코드 수를 초과합니다):
        
        Records | Superclass | Description
        9528 | NORM | Normal ECG
        5486 | MI | Myocardial Infarction
        5250 | STTC | ST/T Change
        4907 | CD | Conduction Disturbance
        2655 | HYP | Hypertrophy
        
        파형 파일은 **1μV/LSB의 해상도와 500Hz의 샘플링 주파수(records500/)에서 16비트 정밀도의 웨이브폼 데이터베이스(WFDB) 포맷으로 저장**됩니다. 사용자의 편의를 위해 샘플링 주파수 100Hz(records100/)의 다운샘플링된 파형 데이터 버전도 공개합니다.
        
        모든 관련 메타데이터는 ecg_id로 식별되는 레코드당 한 행씩 ptbxl_database.csv에 저장됩니다. 여기에는 28개의 열로 분류할 수 있습니다:
        
        1. Identifiers: 각 **기록은 고유한 ecg_id로 식별**됩니다. 해당 **환자는 patient_id**를 통해 인코딩됩니다. **원본 기록(500Hz)**과 기록의 다운샘플링된 버전(100Hz)에 대한 경로는 **filename_hr** 및 filename_lr에 저장됩니다.
        2. 일반 메타데이터: 나이, 성별, 키, 몸무게, 간호사, 사이트, 장치 및 기록_날짜 등의 인구통계 및 기록 메타데이터입니다.
        3. ECG statements: 핵심 구성 요소는 **scp_codes**(형식 문의 항목이 있는 사전으로서의 SCP-ECG statements: **가능성**, 알 수 없는 경우 가능성은 0으로 설정됨) 및 **report(보고서 문자열)**입니다. 추가 필드는 heart_axis, infarction_stadium1, infarction_stadium2, validated_by, second_opinion, initial_autogenerated_report 및 validated_by_human입니다.
        4. 신호 메타데이터: 노이즈(정적 노이즈 및 버스트 노이즈), 기준선 드리프트(기준선_드리프트) 및 **전극_문제와 같은 기타 아티팩트와 같은 신호 품질**. 또한 **추가 수축기 수를 계산하기 위한 추가_비트**와 **활성 심박조율기를 나타내는 신호 패턴을 위한 심박조율기를 제공**합니다.
        5. Cross-validation Folds: recommended 10-fold train-test splits (`strat_fold`) obtained via stratified sampling while respecting patient assignments, i.e. all records of a particular patient were assigned to the same fold. Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. We therefore propose to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
            
            교차 검증 폴드: 환자 할당을 존중하면서 계층화된 샘플링을 통해 얻은 권장 10배 훈련-테스트 분할(strat_fold), 즉 특정 환자의 모든 기록이 동일한 폴드에 할당된 경우입니다. 9번과 10번 폴드의 기록은 적어도 한 번 이상의 사람 평가를 거쳤기 때문에 라벨 품질이 특히 높습니다. 따라서 **폴드 1-8을 훈련 집합으로, 폴드 9를 검증 집합으로, 폴드 10을 테스트 집합으로 사용할 것을 제안합니다**.
            
        
        All information related to the used annotation scheme is stored in a dedicated `scp_statements.csv` that was enriched with mappings to other annotation standards such as AHA, aECGREFID, CDISC and DICOM. We provide additional side-information such as the category each statement can be assigned to (diagnostic, form and/or rhythm). For diagnostic statements, we also provide a proposed hierarchical organization into `diagnostic_class` and `diagnostic_subclass`.
        
        ### ****Usage Notes****
        
        **example_physionet.py**에서는 제안된 훈련-테스트 분할을 사용하여 파형 데이터(numpy-arrays `X_train` and `X_test`)와 레이블(y_train 및 y_test)을 로드하는 방법을 보여주는 최소한의 사용 예제를 제공합니다. 예시에서는 진단 서브클래스 문을 **scp_statements.csv**의 할당을 기반으로 하는 레이블로 사용합니다.
        
    - PTB-XL 데이터셋 요약
        1. **크기 및 구성**: PTB-XL 데이터셋은 18,885명의 환자로부터 취득한 21,837개의 ECG 기록을 포함하며, 각 ECG는 10초 길이입니다.
        2. **주석 및 메타데이터**: 원시 ECG 신호 데이터는 최대 두 명의 심장 전문의에 의해 71가지 다른 ECG 진술로 주석이 달렸으며, 풍부한 메타데이터로 보완되어 있습니다.
        3. **기록 기간 및 장비**: PTB-XL 데이터셋의 원시 신호 데이터는 1989년 10월부터 1996년 6월 사이에 Schiller AG의 장비를 사용하여 수집되었습니다.
        4. **데이터 형식 및 품질**: 데이터셋에서 얻은 중간 비트는 두 가지 상업적 알고리즘으로 처리되었고, 16비트 정밀도, 1μV/LSB의 해상도 및 500Hz의 샘플링 빈도로 WFDB(WaveForm DataBase) 형식으로 저장되었습니다.
        5. **성별 및 연령 범위**: 데이터셋은 성별이 균형을 이루며, 52%가 남성, 48%가 여성이고, 2세에서 95세까지의 넓은 연령 범위를 포함하고 있으며, 중간 연령은 62세입니다.
        6. **독특성**: 이 데이터셋은 PTB에서 같은 장기 프로젝트 동안 수집되었지만, PTB 진단 ECG 데이터베이스와는 기록이 공유되지 않습니다.
        7. **머신 러닝에서의 사용**: PTB-XL 데이터베이스는 그 규모와 데이터 품질 덕분에 ECG 신호 분류를 위한 머신 러닝 알고리즘을 평가하는 데 특히 활용되고 있습니다.
        
    - hr,lr차이와 나이 성별에 대해서
        
        SHP데이터를 살펴보면 설명에 'The sampling frequency is 500 Hz'라고 되어있음. PTB의 심박수 데이터중 hr을 사용.lr을 사용할경우 shp의 다운 샘플링이 필요
        
        심박수 데이타 이외에 나이, 성별만 공통으로 있어서 이외에는 활용하기 어려움
        
        나이, 성별 데이터에는 
        
        PTB에는 300에 해당하는 나이가 293명 있음
        
        PTB성별은 1과 0, SPH성별은 fm으로 되어있음
        
    - ptbxl_database.csv 분석
        
        
        - 컬럼 정보
            1. **`ecg_id`**: ECG 기록의 고유 식별자입니다.
                1. 각 기록에 대한 고유 식별자로, 총 21,799개의 고유한 값이 있습니다. 결측값은 없습니다.
            2. **`patient_id`**: 환자의 고유 식별자입니다.
                1. 환자를 식별하는 번호로, 18,869개의 고유한 값이 있으며, 일부 환자는 여러 개의 기록을 가지고 있습니다.
            3. **`age`**: 환자의 나이입니다.
                1. 환자의 나이로, 89개의 고유한 값이 있습니다. 가장 많은 빈도를 보이는 나이는 67세입니다.
            4. **`sex`**: 환자의 성별을 나타내는 이진 코드입니다 (1은 남성, 0은 여성).
                1. 환자의 성별을 나타내며, 2개의 고유한 값(1은 남성, 0은 여성)이 있습니다. 여성이 약간 더 많은 분포를 보입니다.
            5. **`height`**: 환자의 키입니다. 많은 데이터가 누락되어 있습니다.
                1. 환자의 키로, 77개의 고유한 값이 있으나 대부분의 기록에서 결측값입니다.
            6. **`weight`**: 환자의 몸무게입니다. 이 또한 많은 데이터가 누락되어 있습니다.
                1. 환자의 몸무게로, 127개의 고유한 값이 있으며, 마찬가지로 많은 결측값이 있습니다.
            7. **`nurse`**: 데이터를 기록한 간호사의 식별 번호입니다.
                1. 데이터를 기록한 간호사의 ID로, 12개의 고유한 값이 있습니다.
            8. **`site`**: ECG가 기록된 위치의 식별 번호입니다.
                1. ECG가 기록된 장소의 ID로, 51개의 고유한 값이 있습니다.
            9. **`device`**: ECG 기록에 사용된 장치입니다.
                1. 사용된 ECG 장비로, 11개의 고유한 장비가 있습니다.
            10. **`recording_date`**: ECG 기록 날짜와 시간입니다.
                1. 기록 날짜와 시간으로, 거의 모든 기록이 고유한 시간을 가지고 있습니다.
            11. **`report`: ECG 기록에 대한 의료 보고서 텍스트입니다.**
                1. ECG에 대한 의료 보고서 텍스트로, 9,887개의 고유한 값이 있습니다.
            12. **`scp_codes`: ECG 진단에 대한 SCP-ECG 코드입니다.**
                1. ECG 진단에 대한 SCP-ECG 코드로, 매우 다양한 값을 가지고 있습니다.
            13. **`heart_axis`**: **환자의 심장 축 정보입니다.**
                1. 결측치 다수
                2. 'LAD'는 좌축전위, 'RAD'는 우축전위를 의미합니다. 이는 심장의 전기적 축이 정상 범위를 벗어났음을 나타낼 수 있습니다.
            14. `**infarction_stadium1**`: **심근 경색의 초기 단계 정보입니다.**
                1. 결측치 다수
            15. **`infarction_stadium2`**: **심근 경색의 후기 단계 정보입니다.**
                1. 결측치 다수
            16. **`validated_by`**: 데이터를 검증한 사람의 식별 번호입니다.
                1. 결측치 다수
                2. 'Stadium I'은 초기 단계, 'Stadium II'는 중간 단계, 'Stadium III'은 후기 단계를 의미. 'unknown'은 심근경색의 단계가 알려지지 않았음을 의미
            17. **`second_opinion`**: 두 번째 의견이 있는지 여부를 나타내는 불린값입니다.
                1. 결측치 다수
            18. **`initial_autogenerated_report`**: 초기 자동 생성된 보고서의 유무를 나타내는 불린값입니다.
                1. 결측치 다수
            19. **`validated_by_human`**: 데이터가 인간에 의해 검증되었는지 여부를 나타내는 불린값입니다.
                1. 결측치 다수
            20. **`baseline_drift`**: 기준선 드리프트의 정보입니다.
                1. 결측치 다수
            21. **`static_noise`**: 정적 노이즈의 정보입니다.
                1. 결측치 다수
            22. **`burst_noise`**: 버스트 노이즈의 정보입니다.
                1. 결측치 다수
            23. **`electrodes_problems`**: 전극 문제의 정보입니다.
                1. 결측치 다수
            24. **`extra_beats`**: 추가적인 심장 박동의 정보입니다.
                1. 결측치 다수
            25. **`pacemaker`**: 환자가 페이스메이커를 사용하는지 여부입니다.
                1. 결측치 다수
            26. **`strat_fold`**: 데이터를 분할하기 위한 계층적 접기의 식별 번호입니다.
                1. 결측치 다수
            27. **`filename_lr`**: 저해상도 ECG 기록 파일의 이름입니다.
                1. 결측치 다수
            28. **`filename_hr`**: 고해상도 ECG 기록 파일의 이름입니다.
                1. 결측치 다수
        
        ---
        
        ### Q1. report 컬럼의 텍스트 데이터를 NLP해서 딥러닝에 사용할 수 있는가?
        
        1. **언어 처리**: 보고서가 독일어로 되어 있어, 독일어를 처리할 수 있는 NLP 도구나 라이브러리를 사용하여 의미를 파악하고 필요한 정보를 추출해야 할 수 있습니다.
        2. **텍스트 분류**: 보고서의 텍스트를 분석하여, 예를 들어 심방세동과 같은 특정 진단을 나타내는 키워드나 패턴을 식별할 수 있습니다.
        3. **특징 추출**: 의료 보고서에서 중요한 정보(예: 심방세동, 심근경색 등)를 추출하여, 심전도 신호 데이터와 함께 특징(feature)으로 사용할 수 있습니다.
        4. **의미론적 분석**: 텍스트에서 의미론적 정보를 추출하여, 진단의 중요성이나 확신도를 평가할 수 있습니다.
        5. **정보의 통합**: NLP를 통해 추출한 텍스트 기반 정보를 심전도 신호 데이터와 결합하여, 모델의 입력으로 사용할 수 있습니다.
        
        **구현과정**
        
        1. **텍스트 전처리와 특징 추출**
            - 먼저 **`report`** 컬럼에서 필요한 정보를 추출하기 위해 자연어 처리(NLP) 기술을 사용합니다. 이 과정에서 토큰화, 정규화, 불용어 제거, 어간 추출 등의 전처리 작업이 포함될 수 있습니다.
            - 텍스트에서 특징을 추출하기 위해 TF-IDF, Word2Vec, BERT와 같은 기법을 사용할 수 있습니다.
        2. **다중 입력 모델**
            - 심전도 신호 데이터를 처리하기 위해 CNN, RNN(LSTM, GRU) 또는 Transformer와 같은 신경망 아키텍처를 사용할 수 있습니다.
            - 동시에, 텍스트 데이터를 위한 별도의 NLP 모델을 구성하여, 텍스트에서 추출한 특징을 신경망에 입력할 수 있습니다.
            - 두 모델의 출력을 결합하여 최종 분류 또는 예측을 위한 하나의 모델로 통합할 수 있습니다.
        3. **모델 학습과 최적화**
            - 결합된 모델을 사용하여 분류 작업(예: 심방세동 탐지)에 대한 학습을 진행합니다.
            - 모델의 성능을 최적화하기 위해 하이퍼파라미터 튜닝, 교차 검증, 정규화 기술 등을 적용할 수 있습니다.
        4. **모델 평가**
            - 테스트 데이터셋을 사용하여 모델의 성능을 평가합니다.
            - 정확도, 정밀도, 재현율, F1 점수 등의 지표를 사용하여 모델의 예측 성능을 평가할 수 있습니다.
            
        
    - 신호데이터 (N, 5000) 만들기
        
        # 신호데이터 (N, 5000) 만들기
        
        ```python
        import wfdb
        import numpy as np
        import pandas as pd
        
        # 데이터 프레임 로드
        path = './ptb-xl/ptb-xl/'
        df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        
        # 파일 이름 리스트 변환
        filenames = df['filename_hr'].tolist()
        base_path = './ptb-xl/ptb-xl/'
        
        # Lead I 신호를 저장할 배열 초기화
        lead1_signals = []
        
        # 각 파일에서 Lead I 신호 추출
        for filename in filenames:
            signal, meta = wfdb.rdsamp(base_path + filename)
            lead1 = signal[:, 0]  # Lead I 신호 추출
            lead1_signals.append(lead1)
        
        # 배열을 numpy 배열로 변환
        lead1_signals = np.array(lead1_signals)
        
        # 최종 배열 형태 확인
        print("Final array shape:", lead1_signals.shape)
        ```
        
        10분 동안 돌아가고만 있다…!
        
    - wfdb 대해서 ****
        
        **`wfdb`**는 PhysioNet에서 제공하는 Waveform Database Software Package의 파이썬 래퍼입니다. 이 패키지는 신호 및 의료 데이터에 대한 읽기 및 쓰기 기능을 제공하며, 생체 신호에 대한 전처리 및 분석을 지원합니다. PhysioNet에서는 다양한 생체 신호 데이터를 제공하는데, 이러한 데이터는 의료 및 신호 처리 연구에 사용됩니다.
        
        **`wfdb`** 패키지는 다양한 형식의 신호 데이터를 처리하는 데 사용되며, 대표적으로 PhysioNet의 데이터 포맷 중 하나인 WFDB (WaveForm DataBase) 형식을 다룹니다. WFDB는 생체 신호 데이터를 저장하고 교환하기 위한 표준 형식 중 하나로 널리 사용되고 있습니다.
        
        일반적으로 **`wfdb`**를 사용하면 다음과 같은 작업을 수행할 수 있습니다:
        
        1. **PhysioNet 데이터 읽기:** **`wfdb`**를 사용하여 PhysioNet에서 제공하는 다양한 생체 신호 데이터를 읽을 수 있습니다.
        2. **신호 처리 및 분석:** 읽은 데이터에 대해 다양한 신호 처리 및 분석 작업을 수행할 수 있습니다.
        3. **WFDB 형식으로 데이터 저장:** 데이터를 WFDB 형식으로 저장할 수 있습니다.
        
        예를 들어, **`wfdb`**를 사용하여 PhysioNet에서 제공하는 심전도(ECG) 데이터를 읽고 처리할 수 있습니다.
        
        ```python
        import wfdb
        
        # PhysioNet에서 제공하는 ECG 데이터 읽기
        record = wfdb.rdrecord('100', pb_dir='mitdb')
        
        # 읽은 데이터에 대한 정보 출력
        print(record.__dict__)
        
        # ECG 신호 플로팅
        wfdb.plot_wfdb(record=record, title='ECG Record 100 from MIT-BIH Database')
        ```
        
        이러한 작업은 의료 연구나 의료 신호 처리 분야에서 데이터를 활용할 때 유용합니다. 자세한 내용 및 사용법은 [wfdb](https://wfdb.readthedocs.io/) 공식 문서를 참조하시기를 권장합니다.
        
    - 나이별, 성별별 AFIB
        
        데이터상 심방세동이 아닌사람에 대한 심방세동인 사람의 비율
        
        (=AFIB가 0에 대한 1인 사림의 비율)
        
        300 세의 경우 90세 이상인 경우를 말함
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2058.png)
        
        성별에 따른 비율
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2059.png)
        
    - 데이터행 차이 (질문)
        
        
        ptbxl_database.csv 파일 안에 21799개 행이 존재함.
        
        근데 캐글 코드는 21837개 존재함. 
        
        Q. [캐글 코드](https://www.kaggle.com/code/jraska1/ptb-xl-ecg-1d-convolution-neural-network)(1D CNN) 생성 시기 (단순히 1년 전이라고만 써있음) / 데이터 논문 출간 시기 20221109
        
        - **충원 답변.**
            
            1.0.1버전과 1.0.3 버전에서 환자 수가 다른데, 우리가 사용하게 될 버전은 1.0.3 버전
            
            1.0.3 버전에는 18,869명의 환자로부터 얻은 21,799개의 임상 12-리드 ECG 라고 되어 있습니다. 
            
            ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2060.png)
            
            캐글에서 보셧던 데이터 셋은 
            
            18,885명의 환자로부터 얻은 21,837개의 임상 12-리드 ECG라고 나와 있습니다. 
            
            중간에 버전이 바뀌면서 데이터 셋이 살짝 바뀐 듯 함
            
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2061.png)
        
    - 코드로 해보는 PTB eda
        
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2062.png)
        
        - ptb-xl의 총 심방세동은 1514개이다.
            - 이를 제외한 나머지는 모두 0으로 처리했다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2063.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2064.png)
        
        - 심전도 검사 결과 나이는 60~69세가 가장 많음
        - 그런데 심방세동의 경우 80세 이상의 결과가 가장 많았으며 50세를 기점으로 확 늘어나기 시작
    
- SPH
    - SPH 데이터셋_metadata 설명
        
        [https://www.nature.com/articles/s41597-022-01403-5](https://www.nature.com/articles/s41597-022-01403-5)
        
        **ptb-xl_database.csv 의 경우 21837의 행 (총 18885명의 환자로부터 21837개의 신호가 있습니다. 신호도 10초로 잘려 있음)** 
        
        **metadata.csv 의 경우 25770 행이 있음(기록 길이 스펙트럼이 다양함)**
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2065.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2066.png)
        
        - SPH는 500Hz로만 데이터 수집함. 그래서 PBT에서도 500만 쓰는 듯
        
        ---
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2067.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2068.png)
        
        - 전체 기록 **25770개** 중에 정상 ECG기록이 13905개, 심방세동의 경우 **675**+210+6 = **891개** 이므로 약 3.45%를 차지한다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2069.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2070.png)
        
        - 이 표에서 E의 COUNT는 modifier code 제외하고 카운팅 한듯. (675+99+13)
        
        [표 2](https://www.nature.com/articles/s41597-022-01403-5#Tab2) 에 표시된 것처럼  11개 범주에 걸쳐 44개의 기본 설명이 있습니다(표  [3](https://www.nature.com/articles/s41597-022-01403-5#Tab3) 참조 ). 표에 표시된 기본 진술의 분포는 매우 불균형하며 여러 가지 이유로 많은 기록이 제외되었기 때문에 모집단의 실제 반영으로 간주되어서는 안 됩니다(방법 섹션 참조). 또한 표  [2](https://www.nature.com/articles/s41597-022-01403-5#Tab2) 에는 핵심 진술의 의미를 구체화하는 데 사용되는 15개의 수식어가 있으며 단독으로 사용할 수는 없습니다 [19](https://www.nature.com/articles/s41597-022-01403-5#ref-CR19) . AHA 표준에 따르면 40개 이상의 수정자가 있습니다. 일부 수식어(예: 308 및 310)는 광범위한 핵심 명령문과 함께 사용할 수 있으며 일부 수식어는 특정 범주에만 사용할 수 있습니다. 예를 들어 330-334는 범주 M의 명령문과 쌍을 이루어야 합니다.
        
        ---
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2071.png)
        
        ---
        
        # AHA_statement
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2072.png)
        
        - E, 50의 경우 AHA 코드로 어떻게 표현하는가
            - 정상의 경우 22;23 으로 표현
            - 심방세동의 경우 50 만 쓰거나 50+346, 50+347 형태로 50 숫자가 있으면 모두 포함.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2073.png)
        
        ---
        
        # 환자 나이
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2074.png)
        
        **metadata 파일에는 나이가 90 넘어도 300 처리가 되어있지 않다.**
        
        ---
        
        # 기록 길이
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2075.png)
        
        ---
        
        # ECG 기록당 진술 수 (?)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2076.png)
        
    - AHA_Code (SPH_Guideline)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2077.png)
        
    - 기록 길이 N (metadata.csv)
        
        **FROM. SPH_Guideline**
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2078.png)
        
        데이터 길이가 5000개 = 10초 데이터
        
        → 그래서 N=5000 이 최솟값인 것 같다. 최대는 28000까지도 있음(56초)
        
        → 5000 인 셀이 18842개
        
        → 총 셀이 25770개 이므로 약 73%의 행이 10초 데이터이고 그 외에는 모두 10초보다 길어서 샘플링해야한다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2079.png)
        
        대부분이 15초 이내
        
    - scp_codes의 SR유형
        
        
        df_ptb_lead1.iloc[20].plot()
        
        ![output.png](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/output.png)
        
        df_ptb_lead1.iloc[3330].plot()
        
        ![output2.png](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/output2.png)
        
    - scp_codes컬럼의 다양한 심전도 패턴
        1. NORM: 'Normal', 정상 심전도 결과
        2. SR: 'Sinus Rhythm', 정상 심박동 리듬
        3. NDT: 'Non-diagnostic T-Wave Abnormality', T파 변이가 진단에 부적합
        4. IMI: 'Inferior Myocardial Infarction', 하심근 경색
        5. ABQRS: 'Abnormal QRS Complex', QRS 복합체의 이상
        6. SARRH: 'Severe Arrhythmia with Right Heart', 우심부를 포함한 심부위 심장 이상
        7. LAFB: 'Left Anterior Fascicular Block', 좌전향 섬광 이상
        8. RVH: 'Right Ventricular Hypertrophy', 우심실 비대
        9. LMI: 'Lateral Myocardial Infarction', 측면심근 경색
        10. IPMI: 'Inferoposterior Myocardial Infarction', 하후면심근 경색
        11. PVC: 'Premature Ventricular Contraction', 조기심실 수축
        12. ISCAL: 'Ischemia of the Cardiac Anterior Wall', 심근의 전벽 이상
        13. AFIB: 'Atrial Fibrillation', 심방 세동
        14. CRBBB: 'Complete Right Bundle Branch Block', 우측 번들 브랜치의 완전한 차단
        
    - float16로된 sph 데이터를 float32로 한 이유
        
        sph 데이터가 float16으로 설정이 되어있었다.
        
        32로 바꿔서 실험하니 정규화 계산이 빠르게 잘 된 것을 확인할 수 있었다.
        
        딥러닝 실험에서 float32가 가장 많이 쓰인다고 한다. 
        

# ****[3] 기본이론****

- 불균형
    - Data 불균형
        
        데이터 불균형 문제는 머신러닝 모델의 성능에 큰 영향을 미칠 수 있으며, 특히 의료 데이터와 같은 분야에서는 이러한 문제가 흔히 발생합니다. PTB, SPH 데이터셋에서 AFIB 라벨이 있는 샘플이 6.95%에 불과한 것은 실제로 중요한 불균형 문제를 나타냅니다. 이를 해결하기 위한 몇 가지 접근 방법이 있습니다:
        
        1. **오버샘플링(Over-sampling)**: 소수 클래스의 샘플을 복제하거나 생성하여 데이터셋의 균형을 맞추는 방법입니다. 대표적인 방법으로 SMOTE(Synthetic Minority Over-sampling Technique)가 있으며, 이는 소수 클래스의 샘플을 합성하여 추가합니다.
        2. **언더샘플링(Under-sampling)**: 다수 클래스의 샘플을 줄여서 전체 데이터셋의 균형을 맞추는 방법입니다. 이 방법은 데이터셋의 크기를 줄이게 되므로 정보 손실이 발생할 수 있습니다.
        3. **클래스 가중치(Class Weighting)**: 모델 학습 시 소수 클래스에 더 큰 가중치를 부여하여, 이 클래스의 샘플이 더 큰 영향을 미치도록 하는 방법입니다. 이는 모델이 소수 클래스의 중요성을 인식하도록 돕습니다.
        4. **데이터 증강(Data Augmentation)**: 기존 데이터에 약간의 변형을 주어 새로운 샘플을 생성하는 방법입니다. 예를 들어, 시계열 데이터의 경우 시간 축을 따라 스케일링, 이동, 노이즈 추가 등의 방법이 사용될 수 있습니다.
        5. **앙상블 기법(Ensemble Methods)**: 여러 모델을 결합하여 전체적인 성능을 향상시키는 방법입니다. 소수 클래스에 초점을 맞춘 다양한 모델을 합쳐서 더 균형 잡힌 예측을 도모할 수 있습니다.
        6. **고급 모델 아키텍처**: GAN(Generative Adversarial Networks)이나 Autoencoder와 같은 고급 모델을 사용하여 데이터의 분포를 학습하고, 소수 클래스의 샘플을 생성할 수 있습니다.
        
        각 방법은 장단점이 있으므로, 특정 데이터셋과 문제에 가장 적합한 방법을 선택하는 것이 중요합니다. 실험을 통해 여러 방법을 시험해 보고, 최적의 결과를 도출하는 것이 바람직합니다.
        
    - Data 불균형관련 코드
        
        [https://datascienceschool.net/03 machine learning/14.02 비대칭 데이터 문제.html](https://datascienceschool.net/03%20machine%20learning/14.02%20%EB%B9%84%EB%8C%80%EC%B9%AD%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%AC%B8%EC%A0%9C.html)
        
    
- 평가
    - AUPRC 개념정리
        
        # AUPRC
        
        [https://heeya-stupidbutstudying.tistory.com/entry/ML-모델-평가지표-오차행렬-PRC-ROC-AUC#:~:text=AUPRC (Area under the Precision-Recall Curve)&text=AUPRC는 PRC 아래의,가까울수록 좋은 모델이다](https://heeya-stupidbutstudying.tistory.com/entry/ML-%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C-%EC%98%A4%EC%B0%A8%ED%96%89%EB%A0%AC-PRC-ROC-AUC#:~:text=AUPRC%20(Area%20under%20the%20Precision%2DRecall%20Curve)&text=AUPRC%EB%8A%94%20PRC%20%EC%95%84%EB%9E%98%EC%9D%98,%EA%B0%80%EA%B9%8C%EC%9A%B8%EC%88%98%EB%A1%9D%20%EC%A2%8B%EC%9D%80%20%EB%AA%A8%EB%8D%B8%EC%9D%B4%EB%8B%A4).
        
        **AUPRC는 PRC 아래의 면적 값**으로, 정밀도와 재현율 모두 1에 가까울수록 좋기 때문에 AUPRC의 값도 1에 가까울수록 좋은 모델이다.
        
        **평균 정밀도 (average precision)**
        
        AUPRC와 평균 정밀도 사이에는 미묘한 기술적 차이가 있으나 일반적으로는 PRC 아랫부분의 면적이라고 이해된다
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2080.png)
        
        [https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/](https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/)
        
        AUPRC(정밀도-재현율 곡선 아래 영역)는 긍정적인 예를 찾는 데 많은 관심을 기울이는 문제 설정에서 불균형 데이터에 대한 유용한 성능 지표입니다. 예를 들어, 흉부 엑스레이에서 기흉을 감지하기 위한 분류기를 구축하고 건강한 폐를 기흉 양성으로 잘못 표시하지 않고 모든 기흉을 찾을 수 있도록 하려고 할 수 있습니다.
        
        모델이 완벽한 AUPRC를 달성하면 모델이 실수로 부정적인 사례/건강한 환자를 긍정적인 것으로 표시(완벽한 정밀도)하지 않고 모든 긍정적인 사례/기흉 환자(완벽한 회상)를 찾았음을 의미합니다. "평균 정밀도"는 AUPRC를 계산하는 특별한 방법 중 하나입니다.
        
        PR 곡선의 x축은 재현율이고 y축은 정밀도입니다. 이는 y축이 재현율이고 x축이 FPR인 ROC 곡선과 대조됩니다. 표시된 ROC 곡선과 유사하게 표시된 PR 곡선에서는 결정 임계값이 암시적이며 별도의 축으로 표시되지 않습니다.
        
        - PR 곡선은 왼쪽 위 모서리, 즉 결정 임계값 1에 해당하는 점(재현율 = 0, 정밀도 = 1)에서 시작합니다(모든 예측 확률이 1보다 작기 때문에 모든 예는 음수로 분류됩니다). 출력 값이 가장 큰 예제의 Ground Truth 라벨(양수 또는 음수)이 PR 곡선의 모양에 큰 영향을 미친다는 것입니다.
        - PR 곡선은 오른쪽 아래에서 끝나며 재현율 = 1이고 정밀도가 낮습니다. 이는 결정 임계값 0에 해당합니다(모든 예측 확률이 0보다 크기 때문에 모든 예가 양수로 분류됨). 0에 가까운 재현율에 대한 정밀도 추정치는 분산이 높은 경향이 있습니다.
        - PR 곡선을 생성하는 사이의 점은 1과 0 사이의 다양한 결정 임계값에 대한 정밀도와 재현율을 계산하여 얻습니다. 대략적인 "각도" 곡선의 경우 몇 가지 결정 임계값만 사용합니다. 더 부드러운 곡선을 위해서는 많은 결정 임계값을 사용합니다.
        
        정밀도와 재현율 간의 균형을 살펴보는 이유는 무엇인가요? 모든 것을 긍정적으로 표시한 단순 분류기를 사용하면 완벽한 재현율(그러나 정밀도가 좋지 않음)을 얻을 수 있고, 모든 것을 부정으로 표시한 단순 분류기를 사용하면 완벽한 정밀도(그러나 나쁜 재현율)를 얻을 수 있기 때문에 재현율과 정밀도를 함께 고려하는 것이 중요합니다.
        
        - AUPRC를 계산하기 위해 평균 정밀도를 사용
        
        ```jsx
        auprc = sklearn.metrics.average_precision_score(true_labels, predicted_probs)
        ```
        
        # 요약
        
        - 모델은 실수로 부정적인 예를 긍정적인 것으로 표시(완벽한 정밀도)하지 않고 모든 긍정적인 예(완벽한 재현율)를 찾을 때 완벽한 AUPRC를 달성합니다.
        - AUPRC의 기준선은 양성 비율과 같습니다. 데이터 세트가 8% 암 사례와 92% 건강한 사례로 구성된 경우 기준 AUPRC는 0.08이므로 이 시나리오에서는 0.40의 AUPRC를 얻는 것이 좋습니다!
        - AUPRC는 모델이 긍정적인 예를 올바르게 처리하는 데 많은 관심을 가질 때 가장 유용합니다.
        - AUPRC 계산에는 참음성이 전혀 포함되지 않습니다.
        
        ---
        
        [https://hwi-doc.tistory.com/entry/모델-평가하기-정확도만-높으면-좋은-모델](https://hwi-doc.tistory.com/entry/%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%ED%95%98%EA%B8%B0-%EC%A0%95%ED%99%95%EB%8F%84%EB%A7%8C-%EB%86%92%EC%9C%BC%EB%A9%B4-%EC%A2%8B%EC%9D%80-%EB%AA%A8%EB%8D%B8)
        
        - **PR curve AUC를 사용하는 경우**
        - 데이터가 불균형 할때
        - 양성 클래스를 탐지하는 중요도가 음성 클래스를 탐지하는 중요도보다 높을때 (ex. 암 환자 진단)
        
        - **ROC curve AUC를 사용하는 경우**
        - 데이터가 균형일때
        - 양성 클래스 탐지와 음성 클래스 탐지의 중요도가 비슷할 때 (ex. 개와 고양이 분류)
        
    - 왜 불균형한 데이터셋에 auprc를 사용할까?
        
        # AUROC & AUPRC
        
        ### AUROC
        
        1. **ROC Curve (Receiver Operating Characteristic Curve)**:
            - ROC Curve는 민감도(재현율)와 1-특이도(False Positive Rate)의 관계를 그래프로 나타냅니다.
            - 민감도는 실제 양성 샘플 중 양성으로 올바르게 예측된 비율입니다.
            - 1-특이도는 실제 음성 중 잘못 양성으로 예측된 비율입니다.
        2. **그래프 생성 방법**:
            - 다양한 임계값을 적용하여 민감도와 1-특이도를 계산합니다.
            - 이 두 값을 y축(민감도)과 x축(1-특이도)에 표시하여 그래프를 그립니다.
            - 임계값을 변화시키면서 그려지는 곡선이 ROC Curve입니다.
        3. **AUROC**:
            - AUROC는 ROC Curve 아래의 면적을 의미합니다.
            - 이 값이 클수록 모델이 양성과 음성을 잘 구분한다는 것을 의미합니다.
            - 완벽한 모델의 경우 AUROC는 1이 됩니다.
        
        ### AUPRC
        
        1. **Precision-Recall Curve**:
            - Precision-Recall Curve는 정밀도와 재현율의 관계를 그래프로 나타냅니다.
            - 정밀도는 양성으로 예측된 샘플 중 실제 양성인 비율입니다.
            - 재현율은 실제 양성 샘플 중 양성으로 올바르게 예측된 비율입니다.
        2. **그래프 생성 방법**:
            - 다양한 임계값을 적용하여 정밀도와 재현율을 계산합니다.
            - 이 두 값을 y축(정밀도)과 x축(재현율)에 표시하여 그래프를 그립니다.
            - 임계값을 변화시키면서 그려지는 곡선이 Precision-Recall Curve입니다.
        3. **AUPRC**:
            - AUPRC는 Precision-Recall Curve 아래의 면적을 의미합니다.
            - 이 값이 클수록 모델이 양성 샘플을 정확히 예측하는 능력이 뛰어남을 나타냅니다.
            - 불균형한 데이터셋에서 AUPRC는 특히 중요한 지표가 됩니다.
        
        ### 결론
        
        AUROC는 모델이 양성과 음성을 얼마나 잘 구분하는지를 평가하는 데 사용되고, AUPRC는 모델이 양성 샘플을 얼마나 잘 예측하는지를 평가하는 데 사용됩니다. 불균형한 데이터셋에서는 AUPRC가 더 중요한 정보를 제공할 수 있습니다.
        
        ---
        
        # 불균형한 데이터셋에서
        
        불균형한 데이터셋에서 AUROC (Area Under the Receiver Operating Characteristic Curve)와 AUPRC (Area Under the Precision-Recall Curve)를 성능 지표로 사용하는 것에는 몇 가지 중요한 차이점이 있습니다. 불균형한 데이터셋의 경우, AUPRC가 더 유용한 이유를 설명해드리겠습니다.
        
        ### **AUROC**
        
        - AUROC는 모델이 양성 클래스와 음성 클래스를 얼마나 잘 구분하는지를 측정합니다.
        - AUROC는 민감도(재현율)와 특이도의 관계를 표현합니다.
        - 하지만, 데이터셋이 불균형할 때, AUROC는 음성 클래스(더 많은 클래스)에 대한 성능에 지나치게 영향을 받을 수 있습니다.
        - 예를 들어, 매우 적은 수의 양성 샘플을 가진 데이터셋에서는 대부분을 음성으로 예측해도 높은 AUROC 값을 얻을 수 있습니다.
        
        ### **AUPRC**
        
        - AUPRC는 양성 클래스에 대한 모델의 성능을 보다 정확하게 측정합니다.
        - AUPRC는 정밀도와 재현율의 관계를 표현하며, 특히 양성 클래스가 적은 불균형한 데이터셋에서 유용합니다.
        - 불균형 데이터셋에서는 음성 클래스의 예측이 정밀도에 미치는 영향이 크기 때문에, AUPRC가 모델이 양성 클래스를 얼마나 잘 예측하는지 더 정확하게 반영합니다.
        - AUPRC는 양성 클래스 예측의 정확도와 민감도를 모두 고려하기 때문에, 불균형 데이터셋에서 모델의 성능을 평가하는 데 더 적합한 지표가 됩니다.
        
        ### **왜 AUPRC가 불균형 데이터셋에서 더 유용한가**
        
        - 불균형 데이터셋에서는 양성 샘플의 수가 적기 때문에, 모델이 실제로 양성을 얼마나 잘 예측하는지가 중요한 평가 기준이 됩니다.
        - AUROC는 음성 샘플이 많을 때 과대평가되는 경향이 있지만, AUPRC는 양성 샘플에 대한 예측 성능을 집중적으로 평가합니다.
        - 따라서, 데이터셋이 불균형할 때 AUPRC는 양성 샘플에 대한 모델의 예측 능력을 보다 정확하게 평가하는 지표가 됩니다.
        
        결론적으로, 데이터셋이 불균형한 경우, AUROC보다 AUPRC가 모델 성능을 평가하는 데 더 적합한 지표입니다. AUPRC는 양성 클래스의 예측 정확도와 민감도를 모두 고려하여, 실제로 중요한 양성 예측의 품질을 더 정확하게 반영합니다.
        
        ---
        
        물론이죠, AUROC와 AUPRC의 차이를 보다 구체적인 예시를 통해 설명해드리겠습니다.
        
        ### 데이터셋 예시
        
        - 어떤 질병을 진단하는 모델이 있다고 가정해봅시다.
        - 이 데이터셋에는 총 1000개의 샘플이 있고, 그 중 100개만이 질병 양성(Positive), 나머지 900개는 음성(Negative)입니다. 이는 전형적인 불균형한 데이터셋입니다.
        
        ### 모델 예측 결과
        
        - **모델 A**는 모든 샘플을 음성으로 예측합니다.
        - **모델 B**는 일부 양성을 정확히 예측하지만, 일부 음성 샘플도 양성으로 잘못 예측합니다.
        
        ### 혼동 행렬
        
        |  | 모델 A (전부 음성 예측) | 모델 B (일부 정확 예측) |
        | --- | --- | --- |
        | 실제 양성 | TP: 0, FN: 100 | TP: 50, FN: 50 |
        | 실제 음성 | TN: 900, FP: 0 | TN: 850, FP: 50 |
        
        ### AUROC와 AUPRC 계산
        
        - **모델 A의 AUROC**: 민감도(재현율)가 0이기 때문에 낮음. 하지만, 특이도가 높아서 나쁘지 않은 AUROC 값을 얻을 수 있습니다.
        - **모델 A의 AUPRC**: 양성 예측이 없으므로 매우 낮음.
        - **모델 B의 AUROC**: 민감도와 특이도가 모두 고려되어 괜찮은 AUROC 값을 얻을 수 있습니다.
        - **모델 B의 AUPRC**: 정밀도와 재현율이 모두 고려되므로, 모델 A보다 훨씬 높은 AUPRC 값을 얻을 수 있습니다.
        
        ### 해석
        
        - **AUROC**: 모델 A가 높은 특이도 때문에 나쁘지 않은 AUROC 값을 가질 수 있지만, 실제로 양성 샘플을 하나도 포착하지 못합니다.
        - **AUPRC**: 모델 A는 AUPRC가 매우 낮습니다. 반면, 모델 B는 양성 샘플을 일부 포착하기 때문에 더 높은 AUPRC를 보여줍니다.
        - 이 예시에서 AUROC는 모델 A가 양성 샘플을 전혀 감지하지 못하는 것을 충분히 반영하지 못하지만, AUPRC는 이러한 차이를 더 명확하게 드러냅니다.
        
        ### 결론
        
        불균형한 데이터셋에서는 AUROC가 모델의 성능을 과대평가할 수 있으며, AUPRC가 양성 샘플에 대한 모델의 예측 능력을 더 정확하게 반영합니다. 따라서, 특히 양성 샘플이 중요한 경우 AUPRC가 더 적절한 성능 지표가 됩니다.
        
        ---
        
        죄송합니다, 현재 시스템에서 직접적인 그래프 생성이 어려운 상황입니다. 대신, 혼동 행렬을 설명을 통해 더 자세히 설명드리겠습니다.
        
        ### 모델 A의 혼동 행렬
        
        - 실제 양성: 100개 (모두 FN으로 분류)
        - 실제 음성: 900개 (모두 TN으로 분류)
        - 모델 A는 모든 샘플을 음성으로 예측합니다.
        - 결과적으로, 모든 양성 샘플이 놓치게 되어 FN이 100개가 됩니다. TN은 900개가 되고, TP와 FP는 0입니다.
        
        | 실제 / 예측 | 양성 예측 | 음성 예측 |
        | --- | --- | --- |
        | 실제 양성 | TP: 0 | FN: 100 |
        | 실제 음성 | FP: 0 | TN: 900 |
        
        ### 모델 B의 혼동 행렬
        
        - 실제 양성: 100개 중 50개를 TP로, 50개를 FN으로 분류
        - 실제 음성: 900개 중 50개를 FP로, 850개를 TN으로 분류
        - 모델 B는 일부 양성을 올바르게 예측하지만, 일부 음성 샘플을 잘못 양성으로 예측합니다.
        - 결과적으로, TP가 50개, FP가 50개, FN이 50개, TN이 850개가 됩니다.
        
        | 실제 / 예측 | 양성 예측 | 음성 예측 |
        | --- | --- | --- |
        | 실제 양성 | TP: 50 | FN: 50 |
        | 실제 음성 | FP: 50 | TN: 850 |
        
        ### 해석
        
        - 모델 A는 양성 샘플을 전혀 예측하지 못하므로 재현율이 0입니다. 하지만, 음성 샘플만을 예측하기 때문에 특이도는 높습니다.
        - 모델 B는 양성 샘플 중 절반을 정확히 예측하므로 재현율이 높아집니다. 하지만, 일부 음성 샘플을 잘못 양성으로 분류하므로 정밀도는 낮아집니다.
        - 이 두 모델을 비교했을 때, 모델 A의 AUROC 값은 높을 수 있지만, 실제 중요한 양성 샘플을 전혀 포착하지 못하는 문제가 있습니다. 반면, 모델 B는 AUROC 값이 낮을 수 있지만, 실제 양성 샘플을 어느 정도 포착하는 것이 AUPRC에서 반영됩니다.
        
        이 예시를 통해, 불균형한 데이터셋에서는 AUROC보다 AUPRC가 모델의 성능을 더 정확하게 반영한다는 것을 알 수 있습니다.
        
        ---
        
        # 정밀도가 높은데 재현율이 낮은 경우 → 음성 오류(양성을 음성으로 분류)
        
        네, 정밀도가 높으면서 재현율이 낮은 상황이 충분히 발생할 수 있습니다. 이러한 상황은 모델이 양성(Positive)으로 예측한 경우가 실제로 대부분 양성이지만, 실제 양성인 케이스들 중 많은 부분을 놓치는 경우에 발생합니다.
        
        **예를 들어, 어떤 질병을 진단하는 모델이 있다고 가정해 봅시다. 이 모델이 아주 조심스럽게 양성 판정을 내리기 때문에, 양성으로 판정한 경우의 대부분이 실제로 양성인 상황(높은 정밀도)이지만, 실제 양성인 많은 환자들을 놓치고 음성으로 잘못 판정(낮은 재현율)하는 상황을 생각할 수 있습니다.**
        
        아래는 이러한 상황을 나타내는 혼동 행렬(Confusion Matrix)의 예시입니다:
        
        | 실제 / 예측 | 양성 예측 (Positive) | 음성 예측 (Negative) |
        | --- | --- | --- |
        | 실제 양성 | 20 (TP) | 80 (FN) |
        | 실제 음성 | 5 (FP) | 95 (TN) |
        
        이 경우,
        
        - 정밀도(Precision) = TP / (TP + FP) = 20 / (20 + 5) = 0.80 (높음)
        - 재현율(Recall) 또는 민감도(Sensitivity) = TP / (TP + FN) = 20 / (20 + 80) = 0.20 (낮음)
        
        이 혼동 행렬에서 보듯이, 모델이 양성으로 예측한 경우 대부분 정확하지만(높은 정밀도), 실제 양성인 경우들 중 많은 부분을 놓치고 있습니다(낮은 재현율). 이러한 상황은 특히 질병 진단, 금융 사기 탐지와 같이 오류의 비용이 높은 분야에서 흔히 볼 수 있습니다.
        
        ---
        
        # 정밀도가 낮은데 재현율이 높은 경우 → 양성 오류(음성을 양성으로 분류)
        
        정밀도가 낮으면서 재현율이 높은 경우는 모델이 대부분의 실제 양성(Positive) 사례들을 잘 포착하지만, 많은 음성(Negative) 사례들을 잘못 양성으로 분류하는 상황에서 발생합니다. 즉, 모델이 양성 판정을 너무 자주 내리기 때문에 실제 양성 사례들을 놓치지는 않지만, 그 과정에서 많은 잘못된 양성 판정을 내리는 것입니다.
        
        이러한 상황을 예시로 설명해보겠습니다.
        
        예를 들어, 모델이 이메일을 스팸으로 분류하는 경우를 생각해봅시다. 이 모델이 대부분의 실제 스팸 이메일을 잘 포착하고 있다면 높은 재현율을 가지게 됩니다. 하지만, 정상 이메일 중 상당수도 스팸으로 잘못 분류한다면, 정밀도는 낮아집니다.
        
        아래는 이러한 상황을 나타내는 혼동 행렬(Confusion Matrix)의 예시입니다:
        
        | 실제 / 예측 | 양성 예측 (Positive) | 음성 예측 (Negative) |
        | --- | --- | --- |
        | 실제 양성 | 80 (TP) | 20 (FN) |
        | 실제 음성 | 60 (FP) | 40 (TN) |
        
        이 경우,
        
        - 정밀도(Precision) = TP / (TP + FP) = 80 / (80 + 60) = 0.57 (낮음)
        - 재현율(Recall) 또는 민감도(Sensitivity) = TP / (TP + FN) = 80 / (80 + 20) = 0.80 (높음)
        
        이 혼동 행렬에서 보듯이, 모델은 실제 양성인 대부분의 사례들을 잘 포착하고 있지만(높은 재현율), 많은 잘못된 양성 판정을 내리고 있습니다(낮은 정밀도). 이런 상황은 특히 건강 진단, 범죄 예방 등의 분야에서 흔히 볼 수 있으며, 이러한 경우 모델이 과도한 '양성 오류'를 범하고 있는 것으로 볼 수 있습니다.
        
    - f1
        
        AUPRC가 최대가 되게 하는 임계값과 F1 점수를 최대화하는 임계값은 서로 다를 수 있습니다. AUPRC는 전체 Precision-Recall Curve 아래의 면적을 나타내며, F1 점수는 특정 임계값에서의 정밀도(Precision)와 재현율(Recall)의 조화 평균을 나타냅니다.
        
        AUPRC를 직접적으로 최대화하는 임계값을 찾는 것은 복잡할 수 있으므로, 실제로는 F1 점수를 최대화하는 임계값을 사용하는 것이 일반적입니다. F1 점수는 정밀도와 재현율을 동시에 고려하여, 두 지표 사이의 균형을 맞춘 임계값을 찾는 데 유용합니다. 그러나 이것이 AUPRC를 최대화하는 임계값이라고는 할 수 없습니다.
        
        AUPRC를 최대화하는 임계값을 직접 계산하는 것은 일반적인 머신러닝 라이브러리에서 제공하지 않는 복잡한 과정을 포함할 수 있으며, 대신 F1 점수와 같은 다른 지표를 사용하여 최적의 임계값을 간접적으로 추정하는 방법을 많이 사용합니다.
        
        정밀도(Precision)와 재현율(Recall)을 모두 최대한 높게 유지하려는 목표를 가진다면, F1 점수를 사용하는 것이 가장 합리적인 방법입니다.
        
        F1 점수는 정밀도와 재현율의 조화 평균으로, 두 지표 사이의 균형을 맞추는 데 효과적인 지표입니다. 이는 두 지표 중 하나가 특별히 높거나 낮을 경우 그 영향을 줄이고, 두 지표가 모두 높은 상태를 유지하는 임계값을 찾는 데 도움을 줍니다.
        
        따라서, 모델의 성능을 평가할 때 정밀도와 재현율을 동시에 고려하고자 한다면, F1 점수를 최대화하는 임계값을 찾아 그 값을 기준으로 Confusion Matrix를 그리는 것이 좋은 접근 방법이 될 것입니다.
        
        ```jsx
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score, accuracy_score
        import seaborn as sns
        import torch
        from tqdm import tqdm
        
        # ... [기존 함수들 유지] ...
        
        # AUPRC를 사용하여 최적의 임계값 찾기
        def find_optimal_threshold_by_auprc(y_true, y_pred):
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            # F1 점수를 최대화하는 임계값 찾기
            # F1 점수는 정밀도와 재현율의 조화 평균
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            return optimal_threshold
        
        def test_model(model, test_loader, data_name):
            # ... [기존 코드 유지] ...
        
            # AUPRC를 사용한 최적의 임계값 찾기
            optimal_threshold = find_optimal_threshold_by_auprc(test_targets, test_preds)
            test_accuracy = accuracy_score(test_targets, np.array(test_preds) > optimal_threshold)
            test_loss /= len(test_loader)
            test_auroc = roc_auc_score(test_targets, test_preds)
            test_auprc = average_precision_score(test_targets, test_preds)
        
            print(f'[{data_name} Test] AUROC: {test_auroc:.4f} / AUPRC: {test_auprc:.4f} / Accuracy: {test_accuracy * 100:.2f}% / Loss: {test_loss:.4f}')
            plot_roc_and_prc(test_targets, test_preds, data_name)
            plot_confusion_matrix(test_targets, test_preds, optimal_threshold, data_name)
        
        # ... [나머지 함수들 유지] ...
        
        test_datasets(Custom1DCNN, './save_model')
        ```
        
        ```jsx
        # AUPRC를 사용하여 최적의 임계값 찾기
        def find_optimal_threshold_by_auprc(y_true, y_pred):
            # precision_recall_curve 함수를 사용하여 주어진 예측에 대한 정밀도, 재현율, 임계값을 계산합니다.
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
            # F1 점수를 계산합니다. F1 점수는 2 * (정밀도 * 재현율) / (정밀도 + 재현율)로 정의됩니다.
            # 이는 정밀도와 재현율의 조화 평균을 나타냅니다.
            f1_scores = 2 * (precision * recall) / (precision + recall)
        
            # np.argmax 함수를 사용하여 F1 점수가 최대가 되는 인덱스를 찾습니다.
            optimal_idx = np.argmax(f1_scores)
        
            # 해당 인덱스에 해당하는 임계값을 선택합니다.
            optimal_threshold = thresholds[optimal_idx]
        
            # 계산된 최적의 임계값을 반환합니다.
            return optimal_threshold
        
        # 모델을 테스트하는 함수
        def test_model(model, test_loader, data_name):
            # ... [기존 코드 유지] ...
        
            # AUPRC를 기반으로 계산된 최적의 임계값을 얻기 위해 위에서 정의한 함수를 호출합니다.
            optimal_threshold = find_optimal_threshold_by_auprc(test_targets, test_preds)
        
            # 테스트 데이터셋에 대한 정확도를 계산합니다. np.array(test_preds) > optimal_threshold는 
            # 예측된 확률이 최적의 임계값보다 큰 경우를 True로 변환하여 실제 레이블과 비교합니다.
            test_accuracy = accuracy_score(test_targets, np.array(test_preds) > optimal_threshold)
        
            # 테스트 데이터셋에 대한 평균 손실을 계산합니다.
            test_loss /= len(test_loader)
        
            # 테스트 데이터셋에 대한 AUROC 점수를 계산합니다.
            test_auroc = roc_auc_score(test_targets, test_preds)
        
            # 테스트 데이터셋에 대한 AUPRC 점수를 계산합니다.
            test_auprc = average_precision_score(test_targets, test_preds)
        
            # 계산된 성능 지표들을 출력합니다.
            print(f'[{data_name} Test] AUROC: {test_auroc:.4f} / AUPRC: {test_auprc:.4f} / Accuracy: {test_accuracy * 100:.2f}% / Loss: {test_loss:.4f}')
        
            # ROC 커브와 PRC를 그리는 함수와 혼동 행렬을 그리는 함수를 호출합니다.
            plot_roc_and_prc(test_targets, test_preds, data_name)
            plot_confusion_matrix(test_targets, test_preds, optimal_threshold, data_name)
        
        # ... [나머지 함수들 유지] ...
        
        # 모델 클래스와 모델 파일명을 매개변수로 받아 테스트 데이터셋을 로드하고, 모델을 테스트하는 함수를 호출합니다.
        test_datasets(Custom1DCNN, './save_model')
        ```
        
        `test_model` 함수를 변경하는 이유는, 모델을 평가할 때 사용하는 성능 지표와 혼동 행렬(Confusion Matrix)을 생성할 때 사용하는 임계값의 기준을 일치시키기 위함입니다.
        
        모델을 테스트하면서 주로 관심을 가지는 성능 지표가 AUPRC라 하더라도, 혼동 행렬을 생성할 때 사용하는 임계값은 이 지표와 직접적으로 연관되지 않을 수 있습니다. AUPRC는 모델의 전체 성능을 나타내는 지표이지만, 혼동 행렬은 특정 임계값에서 모델이 어떻게 작동하는지를 보여줍니다.
        
        기존의 `test_model` 함수에서는 Youden Index를 사용하여 임계값을 결정했습니다. Youden Index는 민감도(재현율)와 특이도(1-거짓 양성 비율)의 차이를 최대화하는 임계값을 찾는 방법입니다. 하지만, AUPRC가 중요한 성능 지표로 간주되는 경우, Youden Index로 결정된 임계값은 AUPRC를 최대화하는 데 반드시 최적이라고 할 수 없습니다.
        
        따라서, AUPRC를 중요시하는 상황에서는 혼동 행렬의 임계값도 이와 관련된 기준으로 설정하는 것이 좋습니다. 이를 위해 F1 점수를 최대화하는 임계값을 찾는 방식을 사용하는 것입니다. F1 점수는 정밀도와 재현율의 조화 평균으로, 두 지표 모두를 고려한 균형 잡힌 성능 평가를 가능하게 합니다.
        
        결국, `test_model` 함수에서 임계값을 F1 점수를 최대화하는 방식으로 변경하는 것은 AUPRC를 중시하는 평가 방식과 더 잘 부합하도록 모델의 성능을 평가하고, 혼동 행렬을 생성하기 위한 것입니다.
        
    
- 기타
    - Seed 설정
        
        딥러닝 모델 학습 시 일관된 결과를 얻기 위해 시드(seed)를 설정하는 것이 중요합니다. 이는 모델 초기화, 데이터셋 셔플링, 드롭아웃 등의 무작위성을 제어하기 위함입니다. PyTorch에서 시드를 설정하기 위해 다음과 같은 부분을 코드에 추가할 수 있습니다
        
        1. **NumPy 시드 설정**: 데이터 처리에 NumPy를 사용하는 경우에 필요합니다.
        2. **Python 내장 라이브러리 random의 시드 설정**: 일부 PyTorch 함수들이 Python의 random 모듈을 사용할 수 있습니다.
        3. **PyTorch 시드 설정**: PyTorch의 무작위성을 제어합니다.
        4. **CUDA 시드 설정**: GPU를 사용하는 경우에 필요합니다.
        
        아래는 위의 내용을 기반으로 코드에 시드 설정을 추가한 예시입니다
        
        ```jsx
        import torch
        import numpy as np
        import random
        
        # 시드 설정
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 사용하는 경우
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 기존 코드...
        ```
        

# ****[4] ML****

- ML 종합 정리
    
    # **ML**
    
    ---
    
    ## 요약
    
    ### 전처리
    
                    Normalizer, 데이터 불균형 관련, ~~이상치 제거~~
    
    ### 모델
    
                     RandomForestClassifier
    
    ### 평가지표
    
                    confusion_matrix, AUROC
    
                       
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2081.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2082.png)
    
    ---
    
    ## 전처리
    
    **Normalizer**, - 이유:    **`Normalizer`**는 데이터의 각 특성을 개별적으로 단위 길이로 조정하는 스케일링 기법
    
                                     각 샘플(데이터 포인트)에 대해 벡터의 크기(norm)를 1로 만듭
                                       빠르게 학습할 수 있고 과대적합 확률을 낮출 수 있다.
                                      벡터의 길이가 아니라 데이터의 방향이 중요한 경우 자주 사용한다.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2083.png)
    
                      **상대적 크기의 중요성**: 각 심박수 측정치는 다른 단위나 범위를 가질 수 있습니다. 
    
                                      예를 들어, 심박수와 관련된 특성 중 일부는 백분율(percentage)로 표현되고, 다른 특성은 
    
                                     실제 심박수 측정치일 수 있습니다. 이러한 차이 때문에 각 특성의 중요성이 달라질 수 있습니다. 
    
                                     **`Normalizer`**를 사용하면 각 특성이 단위 길이로 조정되므로, 상대적인 중요성을 조정할 수 있습니다.
    
                       - 실제 모델을 테스팅 해보니 Val, ptb-test, sph-test데이터셋 사이의 결과가 다른 것에비해 신뢰할만한 오차를 보였다.
    
    **데이터 불균형 관련**
    
                       - 사용이유:  데이터가 예측값중 1에 대해서 반응을 안하고 모든값을 0으로만 예측하는 케이스가 생김.
    
                                        실제 PTB-XL 데이터 ~~ 이고 SPH데이터 ~~~ 이라 데이터 불균형 심각.
    
                                         이를 해결하고자 사용.
    
                        오버 샘플링, 언더 샘플링, 오버+언더 샘플링 모두 테스트 해봄.
    
                        결과적으로 —가 가장 좋은 성능.
    
    **이상치 제거** - 이상치 그래프
    
         이상치 제거전, 제거후 성능 향상이 있다.
    
    ## 모델
    
    **RandomForestClassifier**         
    
    이유 : optuna사용( XGBClassifier, RandomForestClassifier, CatBoostClassifier,LGBMClassifier 에 대해서 )
    
           가장 성능 좋은 모델을 선택. 
    
              CatBoostClassifier,LGBMClassifier는 다른 두 모델에 비해 성능이 안나옴
    
              XGBClassifier의 RandomForestClassifier보다 약간 낮은 성능이고 최적의 파라미터를 가져올경우 쉽게 0으로만 예측하게됨.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2084.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2085.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2086.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2087.png)
    
    ## 평가
    
     - 1과 0중 0을 아예 맞추지 않는 문제 발생 auroc는 높지만 confusion_matrix로 살펴보면 True Negative(TN), False Negative(FN),값이 0이나 0에 근접하고 그로인해 Specificity, Precision, F1 Score 값도 0인 평가지표가 나옴.
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2088.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2089.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2090.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2091.png)
    
    데이터 불균형때문이라 보고 데이터 불균형 문제를 해소
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2092.png)
    
    언더 샘플링 - ****Ramdom Under-Sampler, Tomek’s link method, Condensed Nearest Neighbour, One Sided Selection, Edited Nearest Neighbours, Neighbourhood Cleaning Rule****
    
    오버 샘플링 ****RandomOverSampler, ADASYN, SMOTE****
    
    복합 샘플링 ****SMOTE+ENN, SMOTE+Tomek****
    
    가장 좋은 성능은 오버 샘플링 ****ADASYN과 복합 샘플링 SMOTE+Tomek****
    
    ****RandomOverSampler의 경우 auroc는 높지만 대부분 0으로 예측하고 1에 대한 예측도가 떨어짐****
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2093.png)
    
    ****ADASYN과 복합 샘플링 SMOTE+Tomek 비교(위가 AD~) sph에 대한 비교****
    
    **pecision(정밀도)**
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2094.png)
    
    ****ADASYN의 정밀도도0.2****
    
    ****SMOTE+Tomek 정밀도 0.21****
    
    ****SMOTE+Tomek 의 정밀도가 더 좋기에 SMOTE+Tomek  선택****
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2095.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2096.png)
    
    과정 평가 및 추가적인 고도화 작업 계획
    
    - 모델의 성능이 너무 낮거나 너무 높으면 예측값을 0으로만 하는 문제가 발생. 추가적인 고도화에 어려움을 겪고 있으면 모든 실험은 auroc 0.65~ 0.75사이값이 나옴.
    
        
    
    ---
    
    # 
    
- ML에서 시계열, 이진분류 모델(chatGPT)
    
    머신런닝에서 시계열, 이진분류 모델은?
    
    머신러닝에서 시계열 데이터에 대한 이진 분류 모델을 만들 때 일반적으로 사용되는 모델 중 몇 가지가 있습니다. 시계열 데이터의 특성에 따라 모델 선택이 달라질 수 있습니다. 몇 가지 예시는 다음과 같습니다:
    
    1. **시계열 특화 모델:**
        - **ARIMA (AutoRegressive Integrated Moving Average):** 시계열 데이터에 대한 전통적인 통계적 모델 중 하나로, 시계열의 자기 회귀 및 이동 평균 구성 요소를 사용합니다. 주로 시계열 예측에 사용되지만, 이진 분류에도 적용 가능합니다.
        - **Prophet:** Facebook에서 개발한 시계열 예측 라이브러리로, 계절성 및 휴일 효과를 고려하여 간단하게 사용할 수 있습니다.
    2. **머신러닝 모델:**
        - **LSTM (Long Short-Term Memory):** 순환 신경망(RNN)의 한 종류로, 시계열 데이터의 장기 및 단기 의존성을 학습할 수 있습니다.
        - **GRU (Gated Recurrent Unit):** 또 다른 RNN의 변형으로, LSTM보다 더 간단한 구조를 가지고 있으면서도 비슷한 성능을 낼 수 있습니다.
        - **XGBoost, LightGBM, 또는 CatBoost:** 그래디언트 부스팅 알고리즘을 기반으로 하는 트리 기반 모델들은 시계열 데이터에서 강력한 성능을 낼 수 있습니다.
    3. **딥러닝 모델:**
        - **CNN-LSTM 모델:** 1차원 컨볼루션 신경망(CNN)과 LSTM을 결합하여 시계열 데이터의 공간 및 시간적 특징을 모두 학습할 수 있는 모델입니다.
        - **WaveNet:** 생성적인 모델로서 주로 음성 생성에 사용되지만, 시계열 패턴을 감지하고 분류하는 데에도 적용할 수 있습니다.
    
    이 외에도 다양한 모델이 시계열 데이터에 적용될 수 있으며, 데이터의 특성과 목표에 따라 적절한 모델을 선택하는 것이 중요합니다.
    

---

# ****[5] DL****

## (1) **모델 구성 및 실험**

- **모델 선택과 구조**
    - 모델의 선택과 장단점
        
        
        질문 1. 왜 우리는 이 모델을 선택했을까?
        
        일반적으로 시계열 데이터의 분석에는 cnn, rnn,lstm, transformer등의 모델을 사용할수 있다. 
        
        우리는 그중에서 cnn 기반의 1d cnn모델을 사용햇다.
        
        1d cnn의 경우 패턴을 식별하는데 효과적이며 또한 특정 세그먼트의 위치가 관련성이 높지 않는 경우에도 좋을 수 잇다. 
        
        기타 다른 모델의 아키텍처들이 자연어 처리에 주로 사용되며 단어가 가진 위치와 순서가 풀고자하는 문제에 영향을 미치는 한편 시계열 데이터를 기반으로 특정시간대의 특성을 찾아 분류 문제를 해결하고자하는 우리에게 적합했다. 
        
        [https://stats.stackexchange.com/questions/550769/why-cnn-is-suitable-for-time-series-data](https://stats.stackexchange.com/questions/550769/why-cnn-is-suitable-for-time-series-data)
        
        질문 2. 왜 모델의 아키텍처를 이렇게 구성하였는가?
        
        처음 모델을 구성하였을 때에는 모델의 뎁스가 깊지 않았다. 
        
        ```jsx
        # SimpleCNN
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2) 
                self.pool = nn.MaxPool1d(kernel_size=5, stride=1) 
                self.conv2 = nn.Conv1d(32, 64, kernel_size=5) 
                self.fc1 = nn.Linear(64 * 2486, 1000) 
                self.fc2 = nn.Linear(1000, 500)
                self.fc3 = nn.Linear(500, 1)
        
            def forward(self, x):
                x = self.pool(nn.functional.leaky_relu(self.conv1(x)))
                x = self.pool(nn.functional.leaky_relu(self.conv2(x)))
                x = x.view(-1, 64 * 2486)
                x = nn.functional.leaky_relu(self.fc1(x))
                x = nn.functional.leaky_relu(self.fc2(x))
                x = self.fc3(x)
                return x
        ```
        
        그러나 우리 팀의 경우에는 데이터 불균형 문제가 심각했으므로 대부분의 경우 학습 데이터는 맞추지만 실제 예측 데이터는 맞추지 못하는 오버피팅 문제가 빈번하게 발생했다. 
        
        따라서 모델의 뎁스를 깊게 하는 방식으로 성능을 높혀갔다. 
        
        모델의 뎁스가 깊게 구성되면 더 복잡하고 긴 문제를 해결할 수있으나 최적화가 어렵다는 단점이 있다. 
        
        ```python
        class Custom1DCNN(nn.Module):
            def __init__(self):
                super(Custom1DCNN, self).__init__()
        
                # Convolutional Blocks
                self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
                self.bn1 = nn.BatchNorm1d(32)
                self.relu = nn.ReLU()
                self.maxpool1 = nn.MaxPool1d(2)
        
                self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
                self.maxpool2 = nn.MaxPool1d(2)
        
                self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
                self.maxpool3 = nn.MaxPool1d(2)
        
                self.conv4 = nn.Conv1d(64, 64, kernel_size=5)
                self.maxpool4 = nn.MaxPool1d(2)
        
                self.conv5 = nn.Conv1d(64, 128, kernel_size=5)
                self.maxpool5 = nn.MaxPool1d(2)
        
                self.conv6 = nn.Conv1d(128, 128, kernel_size=5)
                self.maxpool6 = nn.MaxPool1d(2)
        
                self.dropout1 = nn.Dropout(0.5)
        
                self.conv7 = nn.Conv1d(128, 256, kernel_size=5)
                self.maxpool7 = nn.MaxPool1d(2)
        
                self.conv8 = nn.Conv1d(256, 256, kernel_size=5)
                self.maxpool8 = nn.MaxPool1d(2)
        
                self.dropout2 = nn.Dropout(0.5)
        
                self.conv9 = nn.Conv1d(256, 512, kernel_size=5)
                self.maxpool9 = nn.MaxPool1d(2)
        
                self.dropout3 = nn.Dropout(0.5)
        
                self.conv10 = nn.Conv1d(512, 512, kernel_size=5)
        
                # Fully Connected Blocks
                self.flatten = nn.Flatten()
        
                self.dense1 = nn.Linear(512, 128)
                self.batch_norm_dense1 = nn.BatchNorm1d(128)  # BatchNorm1d for Dense1
                self.dropout4 = nn.Dropout(0.5)
        
                self.dense2 = nn.Linear(128, 32)
                self.batch_norm_dense2 = nn.BatchNorm1d(32)  # BatchNorm1d for Dense2
        
                self.dense3 = nn.Linear(32, 1)
                
        
            def forward(self, x):
                # Convolutional Blocks
                x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
                x = self.maxpool2(self.relu(self.conv2(x)))
                x = self.maxpool3(self.relu(self.conv3(x)))
                x = self.maxpool4(self.relu(self.conv4(x)))
                x = self.maxpool5(self.relu(self.conv5(x)))
                x = self.maxpool6(self.relu(self.conv6(x)))
                x = self.dropout1(x)
                x = self.maxpool7(self.relu(self.conv7(x)))
                x = self.maxpool8(self.relu(self.conv8(x)))
                x = self.dropout2(x)
                x = self.maxpool9(self.relu(self.conv9(x)))
                x = self.dropout3(x)
                x = self.conv10(x)
        
                # Fully Connected Blocks
                x = self.flatten(x)
                x = self.dropout4(self.relu(self.batch_norm_dense1(self.dense1(x))))
                x = self.relu(self.batch_norm_dense2(self.dense2(x)))
                x = self.dense3(x)
        
                return x
        ```
        
        모델의 설계는 성능 측정, 손실함수 , 최적화 및 하이퍼 파라미터 설정등의 프로세스를 거쳤다. 
        
        이중 하이퍼 파라미터 설정에는 각 레이어의 수 , 학습률, 배치 사이즈, 정규화등이 있었다. 
        
        모델은 총 10개의 컨볼루션 계층을 가졌다. 모델과 완전 연결계층에는 배치 정규화를 넣어 훈련을 가속화 하였으며 활성함수로 relu를 사용하였고 maxpool1d를 이용하여 추출한 특성맵을 다운 샘플링하였다. 
        
        모델의 출력부분에서 출력의 특성은 1이며 이진분류를 위한 시그모이드함수를 사용하는 대신 모댈의 손실함수로  BCEWithLogitsLoss()를 사용하였다. 손실함수 내부에 시그모이드가 존재하여 확률을 계산한다.
        
        모델의 훈련은 adam을 사용하여 진행했다. adam은 sgd의 변형으로 학습률을 역전파하여 최적화를 진행한다. 
        
        기본 학습률은 0.001이었으나 optuna를 사용하는 과정에서 학습률을 최적화 하였으며 작은 학습률을 사용하여 학습의 과정에서 안정성을 유지하였다. 
        
        ### .
        
    - 모델링 구조 개념 정리
        
        [https://wikidocs.net/195419](https://wikidocs.net/195419)
        
        **[머신러닝 파이토치 다루기 기초]**
        
        ### nn.BatchNorm1d
        
        nn.BatchNorm1d는 PyTorch 라이브러리의 하나의 정규화(normalization) 모듈입니다. 특히, 인공 신경망에서 배치 정규화(batch normalization)를 수행하는 데 사용됩니다. 배치 정규화는 입력 데이터를 평균과 표준편차로 정규화하여, 모델이 더 잘 수렴하도록 돕는 방법 중 하나입니다.
        
        nn.BatchNorm1d는 1차원 입력에 대해서만 정규화를 수행하며, 생성자에서 num_features 인자를 받습니다. 이 인자는 입력 데이터의 채널 수를 나타냅니다. 예를 들어, nn.BatchNorm1d(num_features=10)은 10개의 채널을 가지는 입력 데이터에 대해서 배치 정규화를 수행하는 BatchNorm1d 모듈을 생성합니다.
        
        ```python
        torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        ```
        
        - num_features: 입력 데이터의 채널 수를 지정하는 인자입니다. 반드시 지정해야 합니다.
        - eps: 분모에 더해지는 작은 값으로, 0으로 나누는 것을 방지하기 위한 인자입니다. 기본값은 1e-05입니다.
        - momentum: 이전 배치의 평균과 분산값을 얼마나 반영할지를 지정하는 인자입니다. 기본값은 0.1입니다.
        - affine: 정규화된 값을 확대 및 이동시킬지 여부를 지정하는 인자입니다. 기본값은 True입니다.
        - track_running_stats: 배치 정규화의 효과를 추적할지 여부를 지정하는 인자입니다. 기본값은 True입니다.
        
        아래는 nn.BatchNorm1d를 사용한 간단한 코드 예시입니다.
        
        ```python
        import torch.nnas nn
        
        classModel(nn.Module):
        def__init__(self):
                super(Model, self).__init__()
                self.fc1 = nn.Linear(100, 50)
                self.bn = nn.BatchNorm1d(num_features=50)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(50, 10)
        
        defforward(self, x):
                x = self.fc1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.fc2(x)
        return x
        ```
        
        nn.Linear 모듈을 이용하여 입력층과 출력층을 구성하고, nn.BatchNorm1d 모듈을 이용하여 은닉층 출력에 배치 정규화를 적용하고 있습니다. 이렇게 구성된 모델은 입력값을 받아 각 층을 거쳐 출력값을 계산하는데, 은닉층의 출력에 배치 정규화를 적용하여 학습이 더 잘 일어나도록 돕습니다.
        
        ### nn.Conv1d
        
        nn.Conv1d는 파이토치에서 1차원 컨볼루션 레이어를 정의하는 클래스입니다. 1차원 컨볼루션은 입력 데이터의 한 방향(주로 시계열 데이터에서는 시간 축)으로 컨볼루션 연산을 수행합니다.
        
        ```sql
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1, bias=True, padding_mode='zeros')
        ```
        
        - in_channels: 입력 데이터의 채널 개수입니다. 예를 들어, 입력 데이터가 RGB 이미지인 경우 in_channels는 3이 됩니다.
        - out_channels: 출력 데이터의 채널 개수입니다. 이는 컨볼루션 필터의 개수를 의미하며, 출력 데이터가 몇 개의 특징 맵으로 변환되는지를 결정합니다.
        - kernel_size: 컨볼루션 필터(커널)의 크기입니다. 정수 또는 튜플 형태로 지정할 수 있습니다. 예를 들어, kernel_size=3은 3개의 연속된 입력 값에 대해 컨볼루션 연산을 수행하고, kernel_size=(3, 5)는 3개의 연속된 입력 값에 대해 한 방향으로 5개의 컨볼루션 연산을 수행합니다.
        - stride: 컨볼루션 필터의 이동 간격을 지정합니다. 정수 또는 튜플 형태로 지정할 수 있습니다. 예를 들어, stride=1은 한 칸씩 이동하면서 컨볼루션 연산을 수행하고, stride=2는 두 칸씩 이동하면서 컨볼루션 연산을 수행합니다.
        - padding: 입력 데이터에 대해 가상의 패딩을 추가하는 것으로, 컨볼루션 연산의 경계 효과를 조절합니다. 정수 또는 튜플 형태로 지정할 수 있습니다. 예를 들어, padding=1은 입력 데이터에 한 칸의 패딩을 추가하고, padding=(1, 2)는 입력 데이터에 한 방향으로 한 칸의 패딩을 추가하고 다른 방향으로 두 칸의 패딩을 추가합니다.
        - dilation: 컨볼루션 필터 내의 값 사이의 간격을 조절하여, 더 넓은 영역을 감지할 수 있도록 합니다. 정수 또는 튜플 형태로 지정할 수 있습니다.
        - groups: 입력 데이터와 출력 데이터의 채널을 그룹화하여 연산을 수행하는 것으로, 다양한 네트워크 아키텍처를 구성하는 데 사용됩니다.
        - bias: 편향(bias) 사용 여부를 결정하는 불리언 값입니다. 기본값은 True로 편향이 사용되지만, False로 설정하여 편향을 사용하지 않을 수도 있습니다.
        
        ### nn.Flatten()
        
        nn.Flatten()은 PyTorch의 텐서를 1차원으로 평탄화(flatten)하는 클래스입니다. 이는 다차원 텐서를 1차원으로 변환하여 다층 퍼셉트론(MLP) 등의 신경망 레이어에 입력으로 제공할 수 있게 해줍니다.
        
        nn.Flatten()은 입력 텐서를 평탄화하는 작업을 수행하므로, 별도의 인수나 기본값이 없습니다.
        
        예를 들어, 크기가 (batch_size, num_channels, height, width)인 4차원 입력 텐서를 평탄화하여 1차원으로 변환하려면 다음과 같이 사용할 수 있습니다:
        
        ```
        x = torch.randn(batch_size, num_channels, height, width)
        flatten = nn.Flatten()
        x_flatten = flatten(x)
        
        ```
        
        x = x.view(x.size(0), -1)과 같은 결과를 얻을 수 있지만, 코드의 가독성이 향상될 수 있습니다.
        
        ### nn.Linear
        
        nn.Linear는 파이토치에서 사용되는 선형 변환(linear transformation)을 수행하는 클래스로, Fully Connected Layer 또는 Dense Layer라고도 불립니다.
        
        nn.Linear 클래스의 생성자`(__init__)`에는 다음과 같은 인수가 있습니다
        
        - in_features (int): 입력 텐서의 크기. 입력 텐서의 차원(dimension) 또는 특성(feature)의 수입니다.
        - out_features (int): 출력 텐서의 크기. 출력 텐서의 차원(dimension) 또는 특성(feature)의 수입니다.
        - bias (bool, optional): 편향(bias)을 사용할지 여부를 지정합니다. 기본값은 True입니다.
        
        nn.Linear 클래스는 두 개의 행렬 가중치(weight)와 편향(bias)을 학습하며, 입력 텐서를 선형 변환하여 출력 텐서를 생성합니다. 선형 변환은 입력 텐서와 가중치 행렬의 행렬 곱을 계산하고, 편향을 더하는 연산으로 이루어집니다.
        
        nn.Linear 클래스의 예제 코드는 다음과 같습니다:
        
        ```python
        import torch
        import torch.nnas nn
        
        # 입력 텐서의 크기가 10이고 출력 텐서의 크기가 20인 선형 변환을 수행하는 nn.Linear 모듈 생성
        linear = nn.Linear(10, 20)
        
        # 입력 텐서 생성 (크기가 10인 벡터)
        input_tensor = torch.randn(1, 10)
        
        # 선형 변환 수행 (입력 텐서를 출력 텐서로 변환)
        output_tensor = linear(input_tensor)
        
        print("Input Tensor Size: ", input_tensor.size())
        print("Output Tensor Size: ", output_tensor.size())
        
        Input Tensor Size:  torch.Size([1, 10])
        Output Tensor Size:  torch.Size([1, 20])
        
        ```
        
        위의 예제에서는 입력 텐서의 크기가 10이고 출력 텐서의 크기가 20인 nn.Linear 모듈을 생성하고, 입력 텐서를 선형 변환하여 출력 텐서를 생성하는 예제입니다. 출력 텐서의 크기는 nn.Linear의 out_features 인수에 지정한 값인 20과 동일합니다.
        
        ### nn.MaxPool1d
        
        nn.MaxPool1d는 파이토치(PyTorch) 라이브러리에서 제공하는 1차원(Max Pooling) 최대 풀링 연산을 수행하는 클래스입니다. 1차원 신호(예: 오디오, 텍스트)를 처리하는 컨볼루션 신경망(Convolutional Neural Network, CNN)에서 주로 사용됩니다.
        
        Max 풀링은 피처 맵(Feature map)의 공간 차원을 줄이는 역할을 합니다. 이를 통해 컨볼루션 연산을 통해 추출된 특징들을 압축하고, 불필요한 정보를 줄이는 효과를 얻을 수 있습니다. Max 풀링은 입력 피처 맵에서 최대값(maximum value)을 선택하여 출력 피처 맵을 생성합니다.
        
        nn.MaxPool1d는 주요한 매개변수로 다음과 같은 것들을 가지고 있습니다:
        
        - kernel_size: 풀링 윈도우의 크기를 나타내는 정수 값 또는 튜플입니다. 입력 신호에서 추출할 최대값을 결정하는데 사용됩니다. 일반적으로 2 또는 3과 같은 작은 정수 값이 사용됩니다.
        - stride: 풀링 윈도우의 이동 간격을 나타내는 정수 값 또는 튜플입니다. 이 매개변수는 풀링 연산의 겹침(overlapping)을 조절하며, 일반적으로 kernel_size와 같은 값을 사용합니다.(기본값: None으로 kernel_size와 같은 값 설정됨)
        - padding: 입력 신호 주위에 추가할 패딩(padding)의 크기를 나타내는 정수 값 또는 튜플입니다. 패딩은 입력 신호의 경계 부분에서 풀링 윈도우가 넘어갈 때 발생하는 정보 손실을 줄이는 역할을 합니다. 일반적으로 0 또는 1과 같은 작은 값이 사용됩니다.(기본값: 0)
        
        nn.MaxPool1d는 입력 텐서의 크기를 변환하여 출력 텐서를 생성합니다. 입력 텐서의 크기와 풀링 윈도우의 크기, 이동 간격, 패딩의 크기 등에 따라 출력 텐서의 크기가 결정됩니다. 이를 통해 피처 맵의 크기를 다운샘플(downsample)하고, 컨볼루션 신경망에서의 공간적인 계산을 줄여 연산 효율성을 높일 수 있습니다.
        
        ```css
        import torch
        import torch.nn as nn
        
        # 입력 텐서 생성 (배치 크기: 1, 채널: 1, 시퀀스 길이: 10)
        input_tensor = torch.randn(1, 1, 10)
        
        # MaxPool1d 인스턴스 생성
        maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # 최대 풀링 수행
        output_tensor = maxpool(input_tensor)
        
        # 입력 텐서와 출력 텐서의 크기 확인
        print("Input tensor size:", input_tensor.size())
        print("Output tensor size:", output_tensor.size())
        
        Input tensor size: torch.Size([1, 1, 10])
        Output tensor size: torch.Size([1, 1, 5])
        
        ```
        
        위의 예제에서는 입력 텐서의 크기가 (1, 1, 10)이었고, kernel_size를 2로 설정하여 2개의 연속된 값 중 최대값을 선택하는 최대 풀링을 수행했습니다. stride를 2로 설정하여 풀링 윈도우를 2개의 값씩 이동하며 수행하였고, padding은 기본값인 0으로 설정되었습니다. 출력 텐서의 크기는 (1, 1, 5)가 되었음을 확인할 수 있습니다.
        
        ### nn.ReLU
        
        nn.ReLU는 파이토치에서 사용되는 ReLU(Rectified Linear Unit) 활성화 함수를 구현한 클래스입니다.
        
        nn.ReLU 클래스는 입력 텐서의 각 요소에 대해 다음과 같은 수식을 적용하여 활성화 함수를 계산합니다:
        
        ```lua
        ReLU(x) = max(0, x)
        
        ```
        
        여기서 x는 입력 텐서의 값이며, 양수인 경우 그대로 반환되고 음수인 경우 0으로 클리핑됩니다.
        
        nn.ReLU()는 inplace라는 인수가 있습니다. inplace 인수를 사용하면, ReLU 함수의 연산을 "in-place"로 수행하게 됩니다. 이는 입력 텐서의 메모리를 직접 수정하여 연산 속도를 향상시키는 효과가 있습니다.
        
        nn.ReLU()의 inplace 인수의 기본값은 False로 되어 있어서, 기본적으로는 원본 텐서를 수정하지 않고 새로운 텐서를 반환합니다. 그러나 inplace=True로 설정하면, 입력 텐서를 직접 수정하게 됩니다.
        
        예를 들어, 다음과 같이 nn.ReLU()를 사용할 수 있습니다
        
        ```python
        import torch
        import torch.nnas nn
        
        # ReLU 레이어 인스턴스화
        relu = nn.ReLU()  # inplace=False가 기본값
        
        # ReLU 연산 적용
        x = torch.randn(5)
        print('x: ',x)
        y = relu(x)  # 원본 x는 수정되지 않고, 새로운 텐서 y를 반환
        print('after ReLU()')
        print('x: ',x)
        print('y: ',y)
        print('-'*60)
        
        # inplace=True로 설정한 ReLU 연산
        x = torch.randn(5)
        print('x: ',x)
        relu_inplace = nn.ReLU(inplace=True)
        y = relu_inplace(x)  # 원본 x가 직접 수정
        print('after ReLU(inplace=True)')
        print('x: ',x)
        print('y: ',y)
        
        x:  tensor([ 0.2232, -0.1573,  0.7052,  0.9337, -0.5115])
        after ReLU()
        x:  tensor([ 0.2232, -0.1573,  0.7052,  0.9337, -0.5115])
        y:  tensor([0.2232, 0.0000, 0.7052, 0.9337, 0.0000])
        -------------------------------------------------------------
        x:  tensor([ 1.1339, -1.0648, -0.4059,  0.0676, -0.1525])
        after ReLU(inplace=True)
        x:  tensor([1.1339, 0.0000, 0.0000, 0.0676, 0.0000])
        y:  tensor([1.1339, 0.0000, 0.0000, 0.0676, 0.0000])
        
        ```
        
        기본적으로 입력 텐서의 양수 값은 그대로 반환되고, 음수 값은 0으로 클리핑되어 출력 텐서가 생성됩니다.
        
        inplace=True를 사용할 경우, 주의해야 할 점은 원본 텐서가 직접 수정되기 때문에, 연산 이후에 원본 텐서를 사용하는 다른 연산에 영향을 줄 수 있다는 점입니다. 따라서 inplace=True를 사용할 때는 주의하여 사용해야 합니다.
        
        ### nn.LeakyReLU
        
        기본적으로 ReLU 함수와 유사하지만, 입력값이 음수일 때 기울기를 0이 아닌 작은 값으로 유지합니다. 이렇게 하면 ReLU 함수에서 발생하는 "죽은 뉴런(dead neuron)" 문제를 완화할 수 있습니다.
        
        nn.LeakyReLU는 생성자에서 negative_slope 인자를 받습니다. 이 값은 입력값이 음수일 때 사용할 기울기 값을 결정합니다. 보통 0.01이나 0.2와 같은 작은 값이 사용됩니다. 예를 들어, nn.LeakyReLU(negative_slope=0.01)는 입력값이 음수일 때 0.01의 기울기를 가지는 LeakyReLU 함수를 반환합니다.
        
        ```python
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        
        ```
        
        - negative_slope: 음수 기울기 값을 지정하는 인자입니다. 기본값은 0.01입니다.
        - inplace: 연산을 직접 입력 데이터에 수행할지 여부를 지정하는 인자입니다. 기본값은 False입니다.
        
        아래는 nn.LeakyReLU를 사용한 간단한 코드 예시입니다
        
        ```python
        import torch.nnas nn
        
        classModel(nn.Module):
        def__init__(self):
                super(Model, self).__init__()
                self.fc1 = nn.Linear(100, 50)
                self.relu = nn.LeakyReLU(negative_slope=0.01)
                self.fc2 = nn.Linear(50, 10)
        
        defforward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
        return x
        
        ```
        
        위 예제에서는 nn.LeakyReLU를 이용하여 은닉층의 활성화 함수로 사용하고 있습니다. nn.Linear 모듈을 이용하여 입력층과 출력층을 구성하고, nn.LeakyReLU 모듈을 이용하여 은닉층에 적용하고 있습니다. 이렇게 구성된 모델은 입력값을 받아 각 층을 거쳐 출력값을 계산하는데, 은닉층의 활성화 함수로 LeakyReLU를 사용하여 비선형성을 추가합니다. 이렇게 생성된 nn.LeakyReLU 객체는 음수 기울기 값을 0.01로 가지는 LeakyReLU 함수를 생성하며, 입력 데이터의 복사본을 만들어서 연산을 수행합니다. 따라서 inplace 인자를 True로 설정하면, 입력 데이터 자체를 변경하여 연산을 수행할 수 있습니다.
        
        ### nn.BCELoss
        
        torch.nn.BCELoss는 PyTorch 라이브러리에서 제공하는 클래스로, 이진 분류(Binary Classification) 문제에서 사용되는 이진 교차 엔트로피(Binary Cross Entropy) 손실 함수를 구현한 것입니다. BCELoss는 예측값과 타겟값 사이의 교차 엔트로피를 계산하여 손실 값을 얻는데 사용됩니다.
        
        torch.nn.BCELoss의 인수와 기본값은 다음과 같습니다:
        
        ```python
        torch.nn.BCELoss(weight=None, reduction='mean', pos_weight=None)
        
        ```
        
        - weight: 각 클래스에 대한 가중치를 지정하는 인수로, 기본값은 None입니다. 클래스 불균형이 있는 경우에 사용될 수 있습니다.
        - reduction: 손실 값을 어떻게 줄일지를 지정하는 인수로, 기본값은 'mean'입니다. 'sum','none' 옵션을 가지고 있습니다
        - pos_weight: 양성 클래스에 대한 가중치를 지정하는 인수로, 기본값은 None입니다. 양성 클래스의 중요도를 조절할 수 있습니다.
        
        reduction 인수를 통해 손실 값을 어떻게 계산할지를 조절할 수 있습니다. 기본값인 'mean'은 교차 엔트로피 값을 평균으로 나누어 손실 값을 계산하는 것을 의미합니다. pos_weight 인수를 통해 양성 클래스에 대한 가중치를 지정할 수도 있습니다.
        
        ---
        
        ### Adam
        
        파이토치에서 제공하는 Adam은 Adam(Adaptive Moment Estimation) 현재 가장 널리 사용되는 옵티마이저로, 이전 그래디언트의 지수적인 이동 평균을 사용하여 학습률을 조절하는 방식으로 모델을 업데이트합니다. Adam은 경사 하강법(Gradient Descent) 알고리즘을 기반으로 하면서도, 모멘텀 및 학습률 감소와 같은 개선된 기능을 추가한 최적화 알고리즘입니다.
        
        torch.optim.Adam 클래스를 사용하여 Adam 옵티마이저를 생성할 수 있습니다. 주요 인수들은 다음과 같습니다:
        
        ```csharp
        torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        ```
        
        - params: 최적화할 모델의 파라미터들을 전달합니다.
        - lr: 학습률(learning rate)로, 기본값은 0.001입니다.
        - betas: Adam 알고리즘에서 사용되는 두 개의 모멘텀 계수(beta1, beta2)를 튜플 형태로 전달합니다. 기본값은 (0.9, 0.999)입니다.
        - eps: 분모를 보호하기 위한 작은 값(epsilon)으로, 기본값은 1e-08입니다.
        - weight_decay: 가중치 감소(L2 정규화)를 적용하는데 사용되는 가중치 감소 계수(weight decay coefficient)로, 기본값은 0입니다.
        - amsgrad: AMSGrad 알고리즘을 사용할지 여부를 결정하는 불리언 값으로, 기본값은 False입니다.
        
        torch.optim.Adam 클래스는 경사 하강법 알고리즘에 Adam의 개선된 기능을 추가하여 최적화를 수행합니다. Adam은 학습률을 자동으로 조절하고, 모멘텀을 활용하여 이전의 경사 정보를 이용하여 더 빠르게 수렴할 수 있는 등의 장점이 있습니다. 사용자는 필요에 따라 인수들의 값을 조정하여 Adam 옵티마이저를 세밀하게 제어할 수 있습니다.
        
        ### AdamW
        
        torch.optim.AdamW는 AdamW (Adam with Weight Decay) 옵티마이저의 구현체로, 파이토치에서 제공되는 옵티마이저 중 하나입니다. AdamW는 Adam 옵티마이저의 변형으로, 가중치 감쇠(weight decay)를 적용하는 것이 특징입니다. 가중치 감쇠는 모델의 가중치를 감소시킴으로써 모델의 복잡성을 제어하고, 오버피팅(overfitting)을 완화하는 효과가 있습니다.
        
        torch.optim.AdamW 함수의 기본 형태와 인수들의 기본값은 다음과 같습니다:
        
        ```csharp
        torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        ```
        
        - params: 최적화할 파라미터들의 iterable
        - lr: 학습률(learning rate), 기본값은 0.001
        - betas: 감마 값들 (beta1, beta2)로 이루어진 튜플, 기본값은 (0.9, 0.999)
        - eps: 분모를 0으로 나누는 것을 방지하기 위한 작은 상수값, 기본값은 1e-08
        - weight_decay: 가중치 감쇠(L2 정규화) 계수, 기본값은 0
        - amsgrad: AMSGrad 알고리즘을 사용할지 여부, 기본값은 False
        
        torch.optim.AdamW는 Adam 옵티마이저의 변형으로, 가중치 감쇠를 적용하여 모델의 일반화 성능을 향상시키는데 사용될 수 있습니다. 그러나 데이터셋과 모델의 특성에 따라 다양한 옵티마이저를 선택하여 실험해보고 최적의 옵티마이저를 선택하는 것이 좋습니다.
        
        ### RMSprop
        
        torch.optim.RMSprop은 파이토치에서 제공하는 최적화 알고리즘 중 하나로, Root Mean Square Propagation의 약자로 RMSprop이라고 불립니다. RMSprop은 딥러닝에서 널리 사용되는 최적화 알고리즘 중 하나로, 주로 순환 신경망(RNN)과 같이 긴 시퀀스 데이터를 다룰 때 사용됩니다.
        
        RMSprop은 경사(gradient)의 크기를 지수 이동 평균(exponential moving average)을 사용하여 조절하며, 경사의 크기에 따라 각각의 파라미터를 업데이트합니다. RMSprop은 이동 평균을 사용하여 경사의 크기를 조절하기 때문에, **이전 기울기의 크기와 현재 기울기의 크기를 비교하여 기울기의 크기가 크게 변하는 경우 더 작은 학습률을 적용하여 안정적인 학습을 할 수 있습니다.**
        
        일반적으로 사용되는 인수와 기본값은 다음과 같습니다:
        
        ```csharp
        torch.optim.RMSprop(params, lr=0.01, momentum=0, alpha=0.99, eps=1e-8, centered=False, weight_decay=0, momentum_decay=0)
        ```
        
        - params (iterable): 최적화할 파라미터들의 iterable. 일반적으로 모델의 model.parameters()를 전달합니다.
        - lr (float, optional, 기본값=0.01): 학습률(learning rate)로, 업데이트 스텝의 크기를 결정합니다. 높은 학습률은 빠른 학습을 가능하게 하지만, 수렴하지 않거나 발산할 수 있습니다.
        - momentum (float, optional, 기본값=0): 모멘텀(momentum)을 사용하여 업데이트에 관성을 부여합니다. 값이 0이면 모멘텀을 사용하지 않습니다.
        - alpha (float, optional, 기본값=0.99): RMSprop에서 이동 평균을 계산할 때 사용되는 계수로, 경사의 크기를 조절합니다. 1에 가까울수록 이동 평균이 빠르게 갱신되어 빠른 학습이 가능하지만, 불안정할 수 있습니다.
        - eps (float, optional, 기본값=1e-8): 분모를 0으로 나누는 것을 방지하기 위한 작은 상수입니다.
        - centered (bool, optional, 기본값=False): True로 설정하면, 중앙화된 RMSprop을 사용하여 업데이트됩니다.
        - weight_decay (float, optional, 기본값=0): 가중치 감쇠(weight decay)를 적용합니다. L2 정규화를 통해 가중치를 규제합니다.
        - momentum_decay (float, optional, 기본값=0): momentum decay를 적용합니다. 일반적으로 0.9 이하의 값으로 설정됩니다.
        
        RMSprop의 주요 특징은 다음과 같습니다:
        
        학습률 감쇠: RMSprop은 경사의 크기에 따라 학습률을 조절하여 안정적인 학습을 수행합니다. 경사의 크기가 크게 변하는 경우 더 작은 학습률을 사용하여 모델이 더 빠르게 수렴하도록 도와줍니다.
        
        이동 평균 사용: RMSprop은 이전 기울기의 크기와 현재 기울기의 크기를 비교하여 경사의 크기를 조절합니다. 이를 통해 각각의 파라미터별로 적절한 학습률을 적용할 수 있습니다.
        
        파라미터별 업데이트: RMSprop은 각각의 파라미터별로 학습률을 조절하여 업데이트합니다. 이를 통해 모델의 각 파라미터가 서로 다른 학습 속도로 업데이트되어 모델의 학습을 더욱 개선할 수 있습니다.
        
        ### SGD
        
        torch.optim.SGD는 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 최적화 알고리즘을 구현한 PyTorch의 클래스입니다. SGD는 딥러닝 모델의 학습에서 가장 기본적으로 사용되는 최적화 알고리즘 중 하나로, 경사 하강법(Gradient Descent)의 확률적인 버전으로, 각 업데이트 스텝마다 무작위로 선택된 일부 샘플(mini-batch)에 대한 손실 함수의 그래디언트를 사용하여 모델을 업데이트합니다.
        
        torch.optim.SGD의 인수들의 기본값은 다음과 같습니다:
        
        ```csharp
        torch.optim.SGD(params, lr=<required>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        
        ```
        
        - params: 최적화할 모델의 파라미터들을 전달합니다.
        - lr: 필수 인수로, 학습률(learning rate)입니다.
        - momentum: 모멘텀(momentum) 값으로, 기본값은 0입니다.
        - dampening: 모멘텀에 적용되는 감쇠(dampening) 값으로, 기본값은 0입니다.
        - weight_decay: 가중치 감소(L2 정규화)를 적용하는데 사용되는 가중치 감소 계수(weight decay coefficient)로, 기본값은 0입니다.
        - nesterov: 불리언 값으로, 네스테로프 모멘텀(Nesterov Momentum)을 사용할지 여부를 결정합니다. 기본값은 False입니다.
        
        기본적으로 momentum과 dampening은 0으로 설정되어 있고, weight_decay와 nesterov는 비활성화 상태입니다. 사용자는 필요에 따라 이러한 기본값들을 조정하여 SGD 알고리즘의 동작을 커스터마이즈할 수 있습니다.
        
        torch.optim.SGD 클래스는 다음과 같이 사용됩니다:
        
        ```
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
        ```
        
        - model.parameters(): 최적화할 모델의 파라미터들을 전달합니다. 이는 모델의 가중치와 편향들을 포함합니다.
        - lr: 학습률(learning rate)로, 가중치를 업데이트할 때 사용되는 스칼라 값입니다. 학습률은 모델이 얼마나 빠르게 수렴할지를 결정하는 하이퍼파라미터로, 너무 작으면 학습이 느리게 되고, 너무 크면 발산할 수 있습니다.
        - momentum (선택적): 모멘텀(momentum) 값을 설정합니다. 모멘텀은 SGD의 변형인 모멘텀 최적화(Momentum Optimization)에서 사용되는 하이퍼파라미터로, 이전 그래디언트의 가중 평균을 사용하여 현재 그래디언트를 업데이트하는 방법입니다. 기본값은 0이며, 값이 0보다 크면 모멘텀 최적화가 적용됩니다. 일반적으로 0.9와 같은 값이 사용됩니다.
        
        torch.optim.SGD 클래스는 다양한 다른 매개변수들을 제공하며, 사용자는 필요에 따라 학습률 감소, 가중치 감소(L2 정규화), 네스테로프 모멘텀(Nesterov Momentum) 등을 설정할 수 있습니다.
        
    - 딥러닝 모델 개념 정리
        
        # **CNN (Convolutional Neural Network)**
        
        - CNN은 주로 이미지 처리에 사용되지만, 시계열 데이터에 대해서도 효과적일 수 있습니다.
        - 시계열 데이터에서는 1D convolution을 사용하여 시간에 따른 패턴을 감지합니다.
        - 이는 데이터의 지역적 패턴을 인식할 수 있게 해주어 시계열에서의 특정 이벤트나 경향성을 잡아내는 데 유용합니다.
        
        # **RNN (Recurrent Neural Network)**
        
        - RNN은 순차적 데이터를 처리하는 데 특화된 네트워크로, 과거 정보를 저장하고 새로운 입력과 결합하여 출력을 생성합니다.
        - 시계열 데이터에 자주 사용되며, 이전의 시점들을 기억함으로써 현재 상태를 예측하는 데 유리합니다.
        - 그러나 장기 의존성 문제(long-term dependencies)로 인해 때때로 성능이 저하될 수 있습니다.
        
        # **LSTM (Long Short-Term Memory)**
        
        - LSTM은 RNN의 한 종류로, 장기 의존성 문제를 해결하기 위해 고안되었습니다.
        - 게이트 메커니즘을 통해 정보를 장기간 저장하거나 버리는 기능을 갖추고 있습니다.
        - 이는 복잡한 시계열 데이터에서 패턴을 학습하는 데 매우 효과적입니다.
        
        # **Transformer**
        
        - Transformer는 주로 자연어 처리(NLP) 분야에서 사용되지만, 'Attention' 메커니즘이라는 강력한 기능 때문에 시계열 분석에서도 사용됩니다.
        - 이 모델은 입력 데이터의 모든 부분에 동시에 주목할 수 있어 시간 순서에 구애받지 않고 중요한 특성을 추출할 수 있습니다.
        - Transformer 기반 모델은 시계열 데이터에서 복잡한 패턴과 관계를 모델링하는 데 유리합니다.
        
        ---
        
        # **CNN이 최선이니?**
        
        CNN이 성능이 가장 좋은지 여부는 심전도(ECG) 데이터의 특성과 탐지하려는 부정맥의 유형에 따라 다를 수 있습니다. CNN은 지역적인 특징을 잘 잡아내고 형태학적 패턴 인식에 강점을 가지고 있기 때문에, 특히 심전도 신호와 같은 시계열 데이터에서 일정한 패턴을 갖는 부정맥을 탐지하는 데 효과적일 수 있습니다.
        
        하지만, CNN이 항상 최상의 성능을 보장하는 것은 아닙니다. 예를 들어, 심전도 데이터에서는 시간적인 정보가 중요한데, 이러한 정보는 RNN이나 LSTM 같은 순환 신경망을 통해 더 잘 처리될 수 있습니다. 또한, Transformer 기반 모델은 긴 시계열 데이터 내의 복잡한 관계를 잡아내는 데 뛰어난 성능을 보일 수 있습니다.
        
        실제로, 여러 딥러닝 아키텍처를 실험해보고 각 모델의 성능을 비교 분석한 뒤, 가장 적합한 모델을 선택하는 것이 중요합니다. 또한, 때때로 다양한 모델을 조합한 앙상블 방법이 단일 모델을 사용하는 것보다 더 좋은 성능을 나타내기도 합니다.
        
        부정맥 탐지와 같은 의료 이미징 문제에 있어서는, 모델의 성능뿐만 아니라 임상적 유효성, 해석 가능성, 실시간 처리 능력 등 다른 요소들도 중요하게 고려되어야 합니다.
        
    - 딥러닝 분석 프로세스
        
        
        1. **데이터 수집 및 준비**
            - 사용 가능한 공개 ECG 데이터셋을 수집합니다.
            - 각 심장 박동의 레이블을 확인하고, 심박세동 관련 부정맥을 포함하는 데이터만 선별합니다.
        2. **데이터 전처리**
            - 노이즈 필터링: ECG 신호에서 노이즈를 제거하기 위해 필터링을 적용합니다.
            - R 피크 탐지: 심장 박동의 기준점을 찾아내기 위해 R 피크 탐지 알고리즘을 사용합니다.
            - 세분화: 각 심장 박동을 개별적인 관측치로 분리하여 모델링에 사용합니다.
            - 정규화: 데이터의 스케일을 맞추기 위해 정규화를 수행합니다.
        3. **라벨링**
            - 데이터에 AF를 나타내는 레이블을 표시합니다. 전문가의 진단을 바탕으로 한 주석이 필요할 수 있습니다.
        4. **특징 추출**
            - 심박 변동성, RR 간격, QRS 복합체의 형태 및 기간 등과 같은 특징을 추출할 수 있습니다.
            - 시간 영역, 주파수 영역, 비선형 특징을 조합하여 사용합니다.
        5. **모델 구축 및 훈련**
            - **CNN:** ECG 데이터의 형태학적 특징을 감지하기 위해 1D CNN을 사용할 수 있습니다. 1D CNN은 지역적 패턴을 잘 감지하여 각 심장 박동의 특징을 잡아낼 수 있습니다
            - **RNN/LSTM:** 순차적 데이터의 시간적 특성을 모델링하기 위해 LSTM을 사용할 수 있습니다. ECG 신호의 시간적인 패턴을 학습하는 데 적합합니다.
            - **Transformer:** 전체 ECG 시퀀스를 동시에 고려하며, 부정맥이 발생할 가능성이 높은 부분에 주목하기 위해 사용할 수 있습니다. Attention 메커니즘은 ECG 신호 내에서 중요한 부분에 더 많은 가중치를 두어 전체적인 시퀀스를 이해하는 데 도움을 줍니다.
            - 앙상블 학습: 여러 모델의 예측을 결합하여 보다 정확한 결과를 얻을 수 있습니다.
        6. **성능 평가**
            - 모델을 테스트 데이터셋에서 평가하여 성능을 측정합니다.
            - 정확도, 정밀도, 재현율, F1 점수 및 ROC 곡선 같은 지표들을 사용하여 모델의 성능을 평가합니다.
        7. **하이퍼파라미터 튜닝 및 최적화**
            - 그리드 서치, 랜덤 서치 또는 베이지안 최적화 방법을 사용하여 모델의 하이퍼파라미터를 조정합니다.
        8. **모델 배포**
            - 모델이 충분히 좋은 성능을 보인다면, 실시간 ECG 데이터에 적용하여 부정맥을 탐지하도록 배포할 수 있습니다.
    - cnn+lstm 의 이유
        
        CNN(Convolutional Neural Network)과 LSTM(Long Short-Term Memory)을 결합하는 것은 이미지나 시계열 데이터에서 공간적 특성과 시간적 순서를 모두 포착하기 위한 전략입니다.
        
        1. **CNN의 역할**: CNN은 데이터에서 공간적 특성을 추출하는데 탁월합니다. 심전도(ECG) 데이터의 경우, CNN은 심장 박동의 각 부분(예: P파, QRS 복합체, T파)에서 패턴을 인식하고 중요한 특성을 추출할 수 있습니다. 이렇게 추출된 특성들은 심장 박동의 구조적인 특성을 반영합니다.
        2. **LSTM의 역할**: LSTM은 시퀀스 데이터에서 시간에 따른 패턴을 인식하는 데 강력한 능력을 가지고 있습니다. 심전도 데이터처럼 연속적인 시간에 따라 변화하는 신호에서, LSTM은 어떤 심장 박동이 이전 박동과 어떻게 관련되는지 학습할 수 있습니다. 즉, 심장의 시간적인 동작을 학습하여, 예를 들어 부정맥과 같이 시간에 따라 변하는 이상 신호를 잡아낼 수 있습니다.
        
        **모델 결합 과정**:
        
        1. **데이터 입력**: ECG 신호가 모델에 입력됩니다.
        2. **Convolutional Layers**: 입력된 ECG 신호는 여러 CNN 계층을 통과하면서 필터(커널)를 사용해 중요한 특성을 추출합니다. 이 과정에서 고주파 잡음이 제거되고, 심장 박동의 중요한 부분이 강조됩니다.
        3. **Max-Pooling**: CNN의 출력은 Max-Pooling을 통해 중요 특성만을 남기고 차원을 축소합니다. 이렇게 함으로써 모델이 중요한 정보에 집중하고, 계산량을 줄이며, 과적합을 방지할 수 있습니다.
        4. **Dropout**: 과적합을 방지하기 위해 일부 뉴런을 임의로 비활성화합니다. 이는 모델이 데이터의 덜 중요한 부분에 지나치게 의존하는 것을 방지합니다.
        5. **LSTM Layers**: CNN에서 추출된 특성은 LSTM 계층으로 전달됩니다. LSTM은 이러한 특성들이 시간에 따라 어떻게 변하는지를 학습합니다. 이는 심전도 신호의 시간적인 연속성과 패턴을 이해하는 데 중요합니다.
        6. **Dense Layers**: LSTM의 출력은 마지막으로 하나 이상의 Dense 계층을 통과하여 최종적으로 분류를 위한 출력을 생성합니다.
        
        이렇게 CNN과 LSTM을 결합함으로써, 모델은 ECG 데이터의 공간적인 특성과 시간적인 패턴을 모두 학습하여, 더욱 정확하게 심장 질환을 분류할 수 있게 됩니다. 이러한 방식으로 부정맥과 같은 복잡한 심장 조건을 식별하는 데 있어서, 더 높은 정확도와 민감도를 달성할 수 있습니다.
        
    - cnn+lstm 첫 실험
        
        [https://arxiv.org/pdf/2209.00988.pdf](https://arxiv.org/pdf/2209.00988.pdf)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2097.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2098.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%2099.png)
        
        ---
        
        깊게 짠 것
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20100.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20101.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20102.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20103.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20104.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20105.png)
        
    - LSTM 인풋 데이터 형태 변경
        
        **`permute`** 메서드는 텐서의 차원을 재배열하는 데 사용됩니다. 이 메서드는 기존 텐서의 차원을 재배열하여 새로운 형태의 텐서를 생성합니다. **`permute`**의 인수는 각 차원의 새로운 순서를 지정합니다.
        
        예를 들어, **`x = x.permute(0, 2, 1)`**에서:
        
        - **`0`**, **`2`**, **`1`**은 각각 원래 텐서의 0번째, 2번째, 1번째 차원이 새로운 텐서에서 어떻게 배치될지를 나타냅니다.
        - 여기서 **`0`**은 원래 텐서의 0번째 차원(보통 배치 크기)이 새로운 텐서의 0번째 차원으로 유지됨을 의미합니다.
        - **`2`**는 원래 텐서의 2번째 차원이 새로운 텐서의 1번째 차원으로 이동함을 나타냅니다.
        - **`1`**은 원래 텐서의 1번째 차원이 새로운 텐서의 2번째 차원으로 이동함을 나타냅니다.
        
        즉, **`x.permute(0, 2, 1)`**를 사용하면, 원래 텐서의 차원 순서가 (배치 크기, 채널 수, 시퀀스 길이)에서 (배치 크기, 시퀀스 길이, 채널 수)로 변경됩니다. 이는 일반적으로 CNN 후 LSTM 계층에 데이터를 전달할 때 사용됩니다. CNN은 (배치 크기, 채널 수, 시퀀스 길이) 형태의 데이터를 처리하지만, LSTM은 (배치 크기, 시퀀스 길이, 특성 수) 형태의 데이터를 요구하기 때문에, 이 차원 변경이 필요합니다.
        
    - lstm
        
        [https://thebook.io/080289/0506/](https://thebook.io/080289/0506/)
        
         
        
        rnn의 경우 가중치가 업데이트 되는 과정에서 기울기가 1보다 작은 값이 계속해서 곱해지기 때문에 기울기 소실 문제 발생 
        
        ⇒ 대안으로 lstm이나 gru 등이 사용됨
        
        lstm의 구조 
        
        잊기게이트(망각게이트) 
        
        - 과거 정보를 어느정도 기억할지를 결정
        - 시그모이드를 사용하여 과거 정보를 확률로 계산
            - 0이면 버리고 1이면 온전히 보존
        - 새로운 입력과 기존 은닉층에서 입력되는 값 두개를 받음
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20106.png)
        
        입력게이트
        
        - 현재 정보 기억을 위해 만들어짐
        - 과거 정보와 현재 정보를 입력받아 시그모이드와 하이퍼볼릭 탄젠트 함수를 기반으로 현재 정보 보존량을 결정
        - 망각게이트와 입력게이트의 차이
            - 망각게이트: 이전 상태 정보를 현재 반영할지 결정
            - 입력게이트 : 현재 상태에 새로운 정보를 반영할지 결정
            
            ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20107.png)
            
        
        출력게이트
        
        셀 
        
        - 각단계의 은닉노드를 메모리 셀이라고함
        - sum(총합)을 사용하여 셀 값을 반영하며 이것으로 기울기 소멸 문제 해결
        - 망각 게이트와 입력게이트의 이ㅓㄴ단계 셀정보를 게산하여 현재단계 셀상태를 업데이트
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20108.png)
        
    
- **하이퍼파라미터 및 옵티마이저**
    - optimizer
        
        심방세동(Atrial Fibrillation) 탐지를 위한 1D CNN 모델에 사용할 수 있는 여러 종류의 최적화 알고리즘을 고려해볼 수 있습니다. 각각의 최적화 알고리즘은 다른 특성과 장단점을 가지고 있어, 모델의 성능과 학습 속도에 영향을 줄 수 있습니다. 여기 몇 가지 대안적인 최적화 알고리즘을 소개합니다
        
        1. **SGD (Stochastic Gradient Descent)**:
            - 가장 기본적인 최적화 알고리즘으로, 각 반복에서 무작위로 선택된 데이터에 대한 그래디언트를 사용하여 모델을 업데이트합니다.
            - **`torch.optim.SGD(model.parameters(), lr=learning_rate)`** 형식으로 사용합니다.
            - 학습률(lr) 조절과 함께 모멘텀(momentum) 매개변수를 추가하여 사용할 수 있습니다.
            - SGD는 때때로 다른 알고리즘보다 느리게 수렴할 수 있지만, 일반화(generalization)에는 종종 더 좋습니다.
        2. **RMSprop**:
            - 학습률을 자동으로 조절하는 알고리즘 중 하나로, 비등방성 데이터(서로 다른 방향으로의 속성이 다른 데이터)에 잘 작동합니다.
            - **`torch.optim.RMSprop(model.parameters(), lr=learning_rate)`** 형식으로 사용합니다.
            - RMSprop는 비교적 빠르게 수렴하며, 불안정한 데이터셋에서도 잘 작동합니다.
        3. **AdamW**:
            - Adam의 변형으로, 가중치 감쇠(weight decay)를 Adam 최적화 과정에 보다 효과적으로 통합합니다.
            - **`torch.optim.AdamW(model.parameters(), lr=learning_rate)`** 형식으로 사용합니다.
            - AdamW는 오버피팅을 억제하고, 모델의 일반화 성능을 개선하는 데 도움이 될 수 있습니다.
        
        심방세동 탐지와 같은 의료 데이터는 복잡하고 민감한 패턴을 가지고 있을 수 있으므로, 다양한 최적화 알고리즘을 실험하여 모델의 성능을 최적화하는 것이 좋습니다. 학습률, 배치 크기, 에포크 수와 같은 다른 하이퍼파라미터와 함께 최적의 조합을 찾는 것이 중요합니다.
        
    - 옵튜나
        
        ```python
        # 사용자 정의 모듈을 임포트합니다.
        from annie_data_prep import create_dataloaders, load_sph_data
        from annie_cnn_model import SimpleCNN, Custom1DCNN, Custom1DCNNWithBatchNormAndDropout
        from annie_cnnlstm_model import CNNLSTMModel
        
        def objective(trial):
            ## 모델 시드 고정 
            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 사용하는 경우
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
        
            # 옵튜나를 이용한 하이퍼파라미터 제안
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            num_epochs = trial.suggest_int('num_epochs', 30, 50)  # Epoch 수를 동적으로 선택
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
        
            # 모델을 GPU로 옮기기
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Custom1DCNN().to(device)
        
            # 손실 함수 및 옵티마이저 설정
            criterion = nn.BCEWithLogitsLoss()
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)
            
            train_loader, val_loader, _ = create_dataloaders(batch_size)
            
        
            # 하이퍼파라미터 출력
            print("-"*40)
            print(">>>>> Trial{trial.number+1} Start...<<<<<")
            print("-"*40)
            print(f"[Learning Rate] : {learning_rate}")
            print(f"[Batch Size] : {batch_size}")
            print(f"[Epochs] : {num_epochs}")  # 수정된 부분
            print(f"[Optimizer] : {optimizer}")
            print("-"*23)
            print(">>>>> Let's GO!!! <<<<<")
            print("-"*23)
        
            #print(f"Hidden Units: {hidden_units}")
        
            # 학습 및 검증 결과 기록을 위한 사전 선언
            model_info = {}
        
            # 학습 및 검증 과정에서의 손실과 정확도 기록
            train_losses = []
            train_accuracies = []
            train_aurocs = []  # 훈련 데이터 AUROC 기록을 위한 리스트
            train_auprcs = []  # 훈련 데이터 AUPRC 기록을 위한 리스트
            val_losses = []
            val_accuracies = []
            val_aurocs = []
            val_auprcs = []  # AUPRC 기록을 위한 리스트 추가
        
            
            #best_auroc = float('-inf')  # 최고 AUROC 기록을 위한 초기값 설정
            best_auprc = float('-inf')  # 최고 AUPRC 기록을 위한 초기값 설정
            best_auprc_info = None  # 최고 AUPRC 값을 가진 모델의 정보를 저장할 변수
        
            epochs_no_improve = 0
            early_stop = False
            patience = 10
        
            for epoch in range(num_epochs):
                # 훈련 루프
                model.train()
                train_loss = 0.0
                train_preds, train_targets = [], []
        
                for inputs, labels in tqdm(train_loader, desc=f"Trial {trial.number+1} - Epoch {epoch+1}/{num_epochs} - Training"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()  # 그래디언트 초기화
        
                    # 순전파 및 역전파
                    outputs = model(inputs)
                    loss = criterion(outputs.view(-1), labels.float())
                    loss.backward()
                    optimizer.step()
        
                    train_loss += loss.item()
                    train_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
                    train_targets.extend(labels.cpu().numpy())
        
                train_loss /= len(train_loader)
                train_accuracy = accuracy_score(train_targets, np.round(train_preds))
                train_auroc = roc_auc_score(train_targets, train_preds)
                train_auprc = average_precision_score(train_targets, train_preds)
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                train_aurocs.append(train_auroc)
                train_auprcs.append(train_auprc)
                print(f">>> [Train] AUROC: {train_auroc:.4f} / AUPRC: {train_auprc:.4f}")
        
                # 검증 루프
                model.eval()
                val_loss, val_preds, val_targets = 0.0, [], []
        
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs.view(-1), labels.float())
                        val_loss += loss.item()
                        val_preds.extend(torch.sigmoid(outputs).view(-1).cpu().detach().numpy())
                        val_targets.extend(labels.cpu().numpy())
        
                val_loss /= len(val_loader)
                val_accuracy = accuracy_score(val_targets, np.round(val_preds))
                val_auroc = roc_auc_score(val_targets, val_preds)
                val_auprc = average_precision_score(val_targets, val_preds)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                val_aurocs.append(val_auroc)
                val_auprcs.append(val_auprc)
        
                print(f">>> [Valid] AUROC: {val_auroc:.4f} / AUPRC: {val_auprc:.4f}")
        
                # 에포크 결과 기록
                epoch_info = {
                    'train_loss': train_loss,
                    'valid_loss': val_loss,
                    'train_accuracy': train_accuracy,  
                    'valid_accuracy': val_accuracy,      
                    'train_auroc': train_auroc,
                    'valid_auroc': val_auroc,
                    'train_auprc': train_auprc,        
                    'valid_auprc': val_auprc
                }
                model_info[epoch + 1] = epoch_info
        
                # 스케줄러 업데이트
                scheduler.step(val_auprc)
        
                # Early Stopping 체크 및 모델 저장
                if val_auprc > best_auprc:
                    best_auprc = val_auprc
                    epochs_no_improve = 0
                    best_auprc_info = epoch_info  # 최고 AUPRC 값을 갱신할 때 정보 저장
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print("Early stopping")
                        break
            
        
            # 전체 학습 과정의 결과를 JSON 파일로 저장
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f, indent=4)
        
            # 최고 AUPRC 값을 가진 모델의 정보 출력
            print("-" * 42)
            print("< Trial {trial.number+1}'s Best Performance>")
            if best_auprc_info is not None:
                items = list(best_auprc_info.items())
                for i, (key, value) in enumerate(items):
                    print(f"[{key}]: {value:.4f}" + (" <- Pick It Up!" if i == len(items) - 1 else ""))
        
            return best_auprc
        
        # 옵튜나 스터디 객체 생성 및 최적화 실행
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        
        # 최적의 하이퍼파라미터 출력
        print("최적의 하이퍼파라미터: {}".format(study.best_trial.params))
        ```
        
        - simplecnn
            
            [I 2023-12-12 15:10:09,210] A new study created in memory with name: no-name-5c32f7eb-9648-4d2f-a48f-c9f1b5577bd8
            
            - ----------------------
            Learning Rate: 1.6424440647025044e-05
            Batch Size: 64
            Epochs: 94
            Optimizer: SGD (
            Parameter Group 0 dampening: 0 differentiable: False foreach: None lr: 1.6424440647025044e-05 maximize: False momentum: 0.9 nesterov: False weight_decay: 0
            )
            -----------------------
            >>>>> Let's GO!!! <<<<<
            -----------------------
            
            Trial 0 - Epoch 1/94 - Training: 100%|██████████| 273/273 [00:16<00:00, 16.99it/s]
            
            >>> [Train] AUROC: 0.4892 / AUPRC: 0.0687
            >>> [Valid] AUROC: 0.5539 / AUPRC: 0.1100
            
            Trial 0 - Epoch 2/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.53it/s]
            
            >>> [Train] AUROC: 0.4880 / AUPRC: 0.0716
            >>> [Valid] AUROC: 0.5540 / AUPRC: 0.1097
            
            Trial 0 - Epoch 3/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.39it/s]
            
            >>> [Train] AUROC: 0.5005 / AUPRC: 0.0788
            >>> [Valid] AUROC: 0.5540 / AUPRC: 0.1096
            
            Trial 0 - Epoch 4/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.05it/s]
            
            >>> [Train] AUROC: 0.4974 / AUPRC: 0.0794
            >>> [Valid] AUROC: 0.5539 / AUPRC: 0.1097
            
            Trial 0 - Epoch 5/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.26it/s]
            
            >>> [Train] AUROC: 0.4983 / AUPRC: 0.0787
            >>> [Valid] AUROC: 0.5541 / AUPRC: 0.1097
            
            Trial 0 - Epoch 6/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.21it/s]
            
            >>> [Train] AUROC: 0.4985 / AUPRC: 0.0786
            >>> [Valid] AUROC: 0.5542 / AUPRC: 0.1098
            
            Trial 0 - Epoch 7/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.24it/s]
            
            >>> [Train] AUROC: 0.4988 / AUPRC: 0.0786
            >>> [Valid] AUROC: 0.5543 / AUPRC: 0.1098
            Epoch 00007: reducing learning rate of group 0 to 1.6424e-06.
            
            Trial 0 - Epoch 8/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.17it/s]
            
            >>> [Train] AUROC: 0.4993 / AUPRC: 0.0788
            >>> [Valid] AUROC: 0.5543 / AUPRC: 0.1098
            
            Trial 0 - Epoch 9/94 - Training: 100%|██████████| 273/273 [00:13<00:00, 20.88it/s]
            
            >>> [Train] AUROC: 0.4994 / AUPRC: 0.0788
            >>> [Valid] AUROC: 0.5543 / AUPRC: 0.1098
            
            Trial 0 - Epoch 10/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.31it/s]
            
            >>> [Train] AUROC: 0.4994 / AUPRC: 0.0789
            >>> [Valid] AUROC: 0.5544 / AUPRC: 0.1099
            
            Trial 0 - Epoch 11/94 - Training: 100%|██████████| 273/273 [00:12<00:00, 21.01it/s]
            
            >>> [Train] AUROC: 0.4995 / AUPRC: 0.0789
            
            [I 2023-12-12 15:12:49,438] Trial 0 finished with value: 0.11001290176988865 and parameters: {'learning_rate': 1.6424440647025044e-05, 'batch_size': 64, 'num_epochs': 94, 'optimizer': 'SGD'}. Best is trial 0 with value: 0.11001290176988865.
            
            >>> [Valid] AUROC: 0.5544 / AUPRC: 0.1099
            Early stopping
            ------------------------------------------
            <Best AUPRC Model's Performance>
            [train_loss]: 0.5127
            [valid_loss]: 0.3681
            [train_accuracy]: 0.9298
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.4892
            [valid_auroc]: 0.5539
            [train_auprc]: 0.0687
            [valid_auprc]: 0.1100 <- Pick It Up!
            
        
        - 1D CNN
            
            [I 2023-12-12 15:31:17,991] A new study created in memory with name: no-name-7406e872-6a30-413f-a386-d4696314da2a
            
            [Learning Rate] : 0.0006197386642866027
            [Batch Size] : 64
            [Epochs] : 49
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.0006197386642866027
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            **Trial 1 - Epoch 1/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.75it/s]**
            
            > [Train] AUROC: 0.5057 / AUPRC: 0.0695
            [Valid] AUROC: 0.5239 / AUPRC: 0.0755
            Trial 1 - Epoch 2/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.24it/s]
            [Train] AUROC: 0.5010 / AUPRC: 0.0715
            [Valid] AUROC: 0.5147 / AUPRC: 0.0716
            Trial 1 - Epoch 3/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.41it/s]
            [Train] AUROC: 0.5057 / AUPRC: 0.0711
            [Valid] AUROC: 0.5577 / AUPRC: 0.0813
            Trial 1 - Epoch 4/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.44it/s]
            [Train] AUROC: 0.4969 / AUPRC: 0.0699
            [Valid] AUROC: 0.6266 / AUPRC: 0.1101
            Trial 1 - Epoch 5/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.23it/s]
            [Train] AUROC: 0.5078 / AUPRC: 0.0707
            [Valid] AUROC: 0.5186 / AUPRC: 0.0838
            Trial 1 - Epoch 6/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.4968 / AUPRC: 0.0702
            [Valid] AUROC: 0.6218 / AUPRC: 0.1040
            Trial 1 - Epoch 7/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.59it/s]
            [Train] AUROC: 0.5019 / AUPRC: 0.0705
            [Valid] AUROC: 0.6369 / AUPRC: 0.1153
            Trial 1 - Epoch 8/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.64it/s]
            [Train] AUROC: 0.5183 / AUPRC: 0.0724
            [Valid] AUROC: 0.6385 / AUPRC: 0.1164
            Trial 1 - Epoch 9/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.64it/s]
            [Train] AUROC: 0.5144 / AUPRC: 0.0718
            [Valid] AUROC: 0.6439 / AUPRC: 0.1233
            Trial 1 - Epoch 10/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.15it/s]
            [Train] AUROC: 0.5051 / AUPRC: 0.0694
            [Valid] AUROC: 0.6513 / AUPRC: 0.1265
            Trial 1 - Epoch 11/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.20it/s]
            [Train] AUROC: 0.5252 / AUPRC: 0.0737
            [Valid] AUROC: 0.6433 / AUPRC: 0.1252
            Trial 1 - Epoch 12/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.38it/s]
            [Train] AUROC: 0.5134 / AUPRC: 0.0751
            [Valid] AUROC: 0.6636 / AUPRC: 0.1311
            Trial 1 - Epoch 13/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.11it/s]
            [Train] AUROC: 0.5270 / AUPRC: 0.0740
            [Valid] AUROC: 0.6388 / AUPRC: 0.1268
            Trial 1 - Epoch 14/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.95it/s]
            [Train] AUROC: 0.5232 / AUPRC: 0.0756
            [Valid] AUROC: 0.6508 / AUPRC: 0.1288
            Trial 1 - Epoch 15/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.45it/s]
            [Train] AUROC: 0.5380 / AUPRC: 0.0786
            [Valid] AUROC: 0.6588 / AUPRC: 0.1283
            Trial 1 - Epoch 16/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.10it/s]
            [Train] AUROC: 0.5298 / AUPRC: 0.0743
            [Valid] AUROC: 0.6662 / AUPRC: 0.1294
            Trial 1 - Epoch 17/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.00it/s]
            [Train] AUROC: 0.5406 / AUPRC: 0.0772
            [Valid] AUROC: 0.6774 / AUPRC: 0.1337
            Trial 1 - Epoch 18/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.04it/s]
            [Train] AUROC: 0.5563 / AUPRC: 0.0810
            [Valid] AUROC: 0.7034 / AUPRC: 0.1471
            Trial 1 - Epoch 19/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.24it/s]
            [Train] AUROC: 0.5658 / AUPRC: 0.0841
            [Valid] AUROC: 0.7243 / AUPRC: 0.1611
            Trial 1 - Epoch 20/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.86it/s]
            [Train] AUROC: 0.5740 / AUPRC: 0.0855
            [Valid] AUROC: 0.7422 / AUPRC: 0.1777
            Trial 1 - Epoch 21/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.33it/s]
            [Train] AUROC: 0.6218 / AUPRC: 0.0997
            [Valid] AUROC: 0.7596 / AUPRC: 0.1817
            Trial 1 - Epoch 22/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.16it/s]
            [Train] AUROC: 0.6305 / AUPRC: 0.1025
            [Valid] AUROC: 0.7718 / AUPRC: 0.1981
            Trial 1 - Epoch 23/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.27it/s]
            [Train] AUROC: 0.6828 / AUPRC: 0.1254
            [Valid] AUROC: 0.7672 / AUPRC: 0.1989
            Trial 1 - Epoch 24/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.05it/s]
            [Train] AUROC: 0.7157 / AUPRC: 0.1453
            [Valid] AUROC: 0.7690 / AUPRC: 0.1711
            Trial 1 - Epoch 25/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.83it/s]
            [Train] AUROC: 0.7377 / AUPRC: 0.1470
            [Valid] AUROC: 0.7768 / AUPRC: 0.1856
            Trial 1 - Epoch 26/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.7604 / AUPRC: 0.1609
            [Valid] AUROC: 0.7911 / AUPRC: 0.1909
            Trial 1 - Epoch 27/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.35it/s]
            [Train] AUROC: 0.7621 / AUPRC: 0.1638
            [Valid] AUROC: 0.7995 / AUPRC: 0.2099
            Trial 1 - Epoch 28/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.52it/s]
            [Train] AUROC: 0.7679 / AUPRC: 0.1757
            [Valid] AUROC: 0.8116 / AUPRC: 0.2375
            Trial 1 - Epoch 29/49 - Training:  97%|█████████▋| 264/273 [00:07<00:00, 35.06it/s]
            > 
            
            ---
            
            **[Learning Rate] : 0.0004178442199935877**
            **[Batch Size] : 64
            [Epochs] : 36**
            **[Optimizer] : Adam (**
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            ...
            )
            
            **Trial 5 - Epoch 36/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.62it/s]**
            
            > [Train] AUROC: 0.9980 / AUPRC: 0.9856
            [I 2023-12-12 15:56:40,699] Trial 4 finished with value: 0.9213366982167329 and parameters: {'learning_rate': 0.0004178442199935877, 'batch_size': 64, 'num_epochs': 36, 'optimizer': 'Adam'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.9914 / AUPRC: 0.9213
            > 
            
            <Best AUPRC Model's Performance>
            [train_loss]: 0.0198
            [valid_loss]: 0.0727
            [train_accuracy]: 0.9951
            [valid_accuracy]: 0.9761
            **[train_auroc]: 0.9980
            [valid_auroc]: 0.9914
            [train_auprc]: 0.9856
            [valid_auprc]: 0.9213 <- Pick It Up!**
            
            **48분 소요**
            
            ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20109.png)
            
            ## [I 2023-12-12 15:31:17,991] A new study created in memory with name: no-name-7406e872-6a30-413f-a386-d4696314da2a
            
            ## [Learning Rate] : 0.0006197386642866027
            [Batch Size] : 64
            [Epochs] : 49
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.0006197386642866027
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 1 - Epoch 1/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.75it/s]
            
            > [Train] AUROC: 0.5057 / AUPRC: 0.0695
            [Valid] AUROC: 0.5239 / AUPRC: 0.0755
            Trial 1 - Epoch 2/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.24it/s]
            [Train] AUROC: 0.5010 / AUPRC: 0.0715
            [Valid] AUROC: 0.5147 / AUPRC: 0.0716
            Trial 1 - Epoch 3/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.41it/s]
            [Train] AUROC: 0.5057 / AUPRC: 0.0711
            [Valid] AUROC: 0.5577 / AUPRC: 0.0813
            Trial 1 - Epoch 4/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.44it/s]
            [Train] AUROC: 0.4969 / AUPRC: 0.0699
            [Valid] AUROC: 0.6266 / AUPRC: 0.1101
            Trial 1 - Epoch 5/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.23it/s]
            [Train] AUROC: 0.5078 / AUPRC: 0.0707
            [Valid] AUROC: 0.5186 / AUPRC: 0.0838
            Trial 1 - Epoch 6/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.4968 / AUPRC: 0.0702
            [Valid] AUROC: 0.6218 / AUPRC: 0.1040
            Trial 1 - Epoch 7/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.59it/s]
            [Train] AUROC: 0.5019 / AUPRC: 0.0705
            [Valid] AUROC: 0.6369 / AUPRC: 0.1153
            Trial 1 - Epoch 8/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.64it/s]
            [Train] AUROC: 0.5183 / AUPRC: 0.0724
            [Valid] AUROC: 0.6385 / AUPRC: 0.1164
            Trial 1 - Epoch 9/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.64it/s]
            [Train] AUROC: 0.5144 / AUPRC: 0.0718
            [Valid] AUROC: 0.6439 / AUPRC: 0.1233
            Trial 1 - Epoch 10/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.15it/s]
            [Train] AUROC: 0.5051 / AUPRC: 0.0694
            [Valid] AUROC: 0.6513 / AUPRC: 0.1265
            Trial 1 - Epoch 11/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.20it/s]
            [Train] AUROC: 0.5252 / AUPRC: 0.0737
            [Valid] AUROC: 0.6433 / AUPRC: 0.1252
            Trial 1 - Epoch 12/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.38it/s]
            [Train] AUROC: 0.5134 / AUPRC: 0.0751
            [Valid] AUROC: 0.6636 / AUPRC: 0.1311
            Trial 1 - Epoch 13/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.11it/s]
            [Train] AUROC: 0.5270 / AUPRC: 0.0740
            [Valid] AUROC: 0.6388 / AUPRC: 0.1268
            Trial 1 - Epoch 14/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.95it/s]
            [Train] AUROC: 0.5232 / AUPRC: 0.0756
            [Valid] AUROC: 0.6508 / AUPRC: 0.1288
            Trial 1 - Epoch 15/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.45it/s]
            [Train] AUROC: 0.5380 / AUPRC: 0.0786
            [Valid] AUROC: 0.6588 / AUPRC: 0.1283
            Trial 1 - Epoch 16/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.10it/s]
            [Train] AUROC: 0.5298 / AUPRC: 0.0743
            [Valid] AUROC: 0.6662 / AUPRC: 0.1294
            Trial 1 - Epoch 17/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.00it/s]
            [Train] AUROC: 0.5406 / AUPRC: 0.0772
            [Valid] AUROC: 0.6774 / AUPRC: 0.1337
            Trial 1 - Epoch 18/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.04it/s]
            [Train] AUROC: 0.5563 / AUPRC: 0.0810
            [Valid] AUROC: 0.7034 / AUPRC: 0.1471
            Trial 1 - Epoch 19/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.24it/s]
            [Train] AUROC: 0.5658 / AUPRC: 0.0841
            [Valid] AUROC: 0.7243 / AUPRC: 0.1611
            Trial 1 - Epoch 20/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.86it/s]
            [Train] AUROC: 0.5740 / AUPRC: 0.0855
            [Valid] AUROC: 0.7422 / AUPRC: 0.1777
            Trial 1 - Epoch 21/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.33it/s]
            [Train] AUROC: 0.6218 / AUPRC: 0.0997
            [Valid] AUROC: 0.7596 / AUPRC: 0.1817
            Trial 1 - Epoch 22/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.16it/s]
            [Train] AUROC: 0.6305 / AUPRC: 0.1025
            [Valid] AUROC: 0.7718 / AUPRC: 0.1981
            Trial 1 - Epoch 23/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.27it/s]
            [Train] AUROC: 0.6828 / AUPRC: 0.1254
            [Valid] AUROC: 0.7672 / AUPRC: 0.1989
            Trial 1 - Epoch 24/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.05it/s]
            [Train] AUROC: 0.7157 / AUPRC: 0.1453
            [Valid] AUROC: 0.7690 / AUPRC: 0.1711
            Trial 1 - Epoch 25/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.83it/s]
            [Train] AUROC: 0.7377 / AUPRC: 0.1470
            [Valid] AUROC: 0.7768 / AUPRC: 0.1856
            Trial 1 - Epoch 26/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.7604 / AUPRC: 0.1609
            [Valid] AUROC: 0.7911 / AUPRC: 0.1909
            Trial 1 - Epoch 27/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.35it/s]
            [Train] AUROC: 0.7621 / AUPRC: 0.1638
            [Valid] AUROC: 0.7995 / AUPRC: 0.2099
            Trial 1 - Epoch 28/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.52it/s]
            [Train] AUROC: 0.7679 / AUPRC: 0.1757
            [Valid] AUROC: 0.8116 / AUPRC: 0.2375
            Trial 1 - Epoch 29/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.41it/s]
            [Train] AUROC: 0.7900 / AUPRC: 0.1928
            [Valid] AUROC: 0.8132 / AUPRC: 0.2025
            Trial 1 - Epoch 30/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.68it/s]
            [Train] AUROC: 0.8002 / AUPRC: 0.2009
            [Valid] AUROC: 0.8268 / AUPRC: 0.2132
            Trial 1 - Epoch 31/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.45it/s]
            [Train] AUROC: 0.8120 / AUPRC: 0.2201
            [Valid] AUROC: 0.8345 / AUPRC: 0.2174
            Trial 1 - Epoch 32/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.71it/s]
            [Train] AUROC: 0.8199 / AUPRC: 0.2178
            [Valid] AUROC: 0.8433 / AUPRC: 0.2282
            Trial 1 - Epoch 33/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.53it/s]
            [Train] AUROC: 0.8311 / AUPRC: 0.2337
            [Valid] AUROC: 0.8484 / AUPRC: 0.2265
            Trial 1 - Epoch 34/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.59it/s]
            [Train] AUROC: 0.8393 / AUPRC: 0.2518
            [Valid] AUROC: 0.8607 / AUPRC: 0.2443
            Trial 1 - Epoch 35/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.09it/s]
            [Train] AUROC: 0.8548 / AUPRC: 0.2666
            [Valid] AUROC: 0.8783 / AUPRC: 0.2660
            Trial 1 - Epoch 36/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.15it/s]
            [Train] AUROC: 0.8668 / AUPRC: 0.2815
            [Valid] AUROC: 0.8854 / AUPRC: 0.2540
            Trial 1 - Epoch 37/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.52it/s]
            [Train] AUROC: 0.8911 / AUPRC: 0.3382
            [Valid] AUROC: 0.8779 / AUPRC: 0.2369
            Trial 1 - Epoch 38/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.29it/s]
            [Train] AUROC: 0.9028 / AUPRC: 0.3574
            [Valid] AUROC: 0.8348 / AUPRC: 0.1849
            Trial 1 - Epoch 39/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.31it/s]
            [Train] AUROC: 0.9160 / AUPRC: 0.3792
            [Valid] AUROC: 0.8927 / AUPRC: 0.2628
            Trial 1 - Epoch 40/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.47it/s]
            [Train] AUROC: 0.9318 / AUPRC: 0.4308
            [Valid] AUROC: 0.7971 / AUPRC: 0.1559
            Trial 1 - Epoch 41/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.42it/s]
            [Train] AUROC: 0.9312 / AUPRC: 0.4553
            [Valid] AUROC: 0.8826 / AUPRC: 0.2420
            Epoch 00041: reducing learning rate of group 0 to 6.1974e-05.
            Trial 1 - Epoch 42/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.11it/s]
            [Train] AUROC: 0.9382 / AUPRC: 0.4675
            [Valid] AUROC: 0.8935 / AUPRC: 0.2606
            Trial 1 - Epoch 43/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.26it/s]
            [Train] AUROC: 0.9423 / AUPRC: 0.4748
            [Valid] AUROC: 0.7429 / AUPRC: 0.1272
            Trial 1 - Epoch 44/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.16it/s]
            [Train] AUROC: 0.9377 / AUPRC: 0.4607
            [Valid] AUROC: 0.8780 / AUPRC: 0.2349
            Trial 1 - Epoch 45/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.03it/s]
            [Train] AUROC: 0.9377 / AUPRC: 0.4781
            [Valid] AUROC: 0.8988 / AUPRC: 0.2705
            Trial 1 - Epoch 46/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.95it/s]
            [Train] AUROC: 0.9443 / AUPRC: 0.4790
            [Valid] AUROC: 0.8831 / AUPRC: 0.2428
            Trial 1 - Epoch 47/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.28it/s]
            [Train] AUROC: 0.9452 / AUPRC: 0.4908
            [Valid] AUROC: 0.8868 / AUPRC: 0.2488
            Trial 1 - Epoch 48/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.42it/s]
            [Train] AUROC: 0.9403 / AUPRC: 0.4854
            [Valid] AUROC: 0.9011 / AUPRC: 0.2749
            Trial 1 - Epoch 49/49 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.47it/s]
            [Train] AUROC: 0.9456 / AUPRC: 0.5004
            [I 2023-12-12 15:37:50,788] Trial 0 finished with value: 0.2749223678673457 and parameters: {'learning_rate': 0.0006197386642866027, 'batch_size': 64, 'num_epochs': 49, 'optimizer': 'SGD'}. Best is trial 0 with value: 0.2749223678673457.
            [Valid] AUROC: 0.6487 / AUPRC: 0.0964
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.1511
            [valid_loss]: 0.2656
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.9403
            [valid_auroc]: 0.9011
            [train_auprc]: 0.4854
            [valid_auprc]: 0.2749 <- Pick It Up!
            
            ## [Learning Rate] : 0.0006674418213852289
            [Batch Size] : 32
            [Epochs] : 44
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.0006674418213852289
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 2 - Epoch 1/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.88it/s]
            
            > [Train] AUROC: 0.5046 / AUPRC: 0.0709
            [Valid] AUROC: 0.4298 / AUPRC: 0.0686
            Trial 2 - Epoch 2/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.82it/s]
            [Train] AUROC: 0.5140 / AUPRC: 0.0721
            [Valid] AUROC: 0.5978 / AUPRC: 0.0959
            Trial 2 - Epoch 3/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.22it/s]
            [Train] AUROC: 0.5004 / AUPRC: 0.0706
            [Valid] AUROC: 0.5950 / AUPRC: 0.1014
            Trial 2 - Epoch 4/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.83it/s]
            [Train] AUROC: 0.5118 / AUPRC: 0.0736
            [Valid] AUROC: 0.6063 / AUPRC: 0.1113
            Trial 2 - Epoch 5/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.09it/s]
            [Train] AUROC: 0.5043 / AUPRC: 0.0696
            [Valid] AUROC: 0.6274 / AUPRC: 0.1194
            Trial 2 - Epoch 6/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.13it/s]
            [Train] AUROC: 0.5252 / AUPRC: 0.0751
            [Valid] AUROC: 0.6288 / AUPRC: 0.1252
            Trial 2 - Epoch 7/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.33it/s]
            [Train] AUROC: 0.5298 / AUPRC: 0.0775
            [Valid] AUROC: 0.6345 / AUPRC: 0.1252
            Trial 2 - Epoch 8/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.86it/s]
            [Train] AUROC: 0.5141 / AUPRC: 0.0712
            [Valid] AUROC: 0.6417 / AUPRC: 0.1265
            Trial 2 - Epoch 9/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.96it/s]
            [Train] AUROC: 0.5136 / AUPRC: 0.0721
            [Valid] AUROC: 0.6497 / AUPRC: 0.1322
            Trial 2 - Epoch 10/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.30it/s]
            [Train] AUROC: 0.5522 / AUPRC: 0.0799
            [Valid] AUROC: 0.6693 / AUPRC: 0.1374
            Trial 2 - Epoch 11/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.09it/s]
            [Train] AUROC: 0.5397 / AUPRC: 0.0772
            [Valid] AUROC: 0.6990 / AUPRC: 0.1428
            Trial 2 - Epoch 12/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.51it/s]
            [Train] AUROC: 0.5806 / AUPRC: 0.0899
            [Valid] AUROC: 0.7259 / AUPRC: 0.1581
            Trial 2 - Epoch 13/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.34it/s]
            [Train] AUROC: 0.6148 / AUPRC: 0.0984
            [Valid] AUROC: 0.7686 / AUPRC: 0.2017
            Trial 2 - Epoch 14/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.93it/s]
            [Train] AUROC: 0.6900 / AUPRC: 0.1298
            [Valid] AUROC: 0.7721 / AUPRC: 0.1872
            Trial 2 - Epoch 15/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.12it/s]
            [Train] AUROC: 0.7424 / AUPRC: 0.1615
            [Valid] AUROC: 0.7825 / AUPRC: 0.1786
            Trial 2 - Epoch 16/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.43it/s]
            [Train] AUROC: 0.7624 / AUPRC: 0.1648
            [Valid] AUROC: 0.7941 / AUPRC: 0.1908
            Trial 2 - Epoch 17/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.49it/s]
            [Train] AUROC: 0.7820 / AUPRC: 0.1830
            [Valid] AUROC: 0.8062 / AUPRC: 0.1827
            Trial 2 - Epoch 18/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.66it/s]
            [Train] AUROC: 0.8014 / AUPRC: 0.2076
            [Valid] AUROC: 0.8194 / AUPRC: 0.1871
            Trial 2 - Epoch 19/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.17it/s]
            [Train] AUROC: 0.8178 / AUPRC: 0.2199
            [Valid] AUROC: 0.8396 / AUPRC: 0.2076
            Trial 2 - Epoch 20/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.82it/s]
            [Train] AUROC: 0.8292 / AUPRC: 0.2272
            [Valid] AUROC: 0.8581 / AUPRC: 0.2225
            Trial 2 - Epoch 21/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.48it/s]
            [Train] AUROC: 0.8636 / AUPRC: 0.2803
            [Valid] AUROC: 0.8548 / AUPRC: 0.2062
            Trial 2 - Epoch 22/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.56it/s]
            [Train] AUROC: 0.9011 / AUPRC: 0.3669
            [Valid] AUROC: 0.9102 / AUPRC: 0.2964
            Trial 2 - Epoch 23/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.53it/s]
            [Train] AUROC: 0.9178 / AUPRC: 0.4038
            [Valid] AUROC: 0.8917 / AUPRC: 0.2572
            Trial 2 - Epoch 24/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.36it/s]
            [Train] AUROC: 0.9314 / AUPRC: 0.4578
            [Valid] AUROC: 0.9073 / AUPRC: 0.2884
            Trial 2 - Epoch 25/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.11it/s]
            [Train] AUROC: 0.9359 / AUPRC: 0.4638
            [Valid] AUROC: 0.7498 / AUPRC: 0.1302
            Trial 2 - Epoch 26/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.69it/s]
            [Train] AUROC: 0.9449 / AUPRC: 0.4944
            [Valid] AUROC: 0.8516 / AUPRC: 0.2016
            Trial 2 - Epoch 27/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.65it/s]
            [Train] AUROC: 0.9459 / AUPRC: 0.5092
            [Valid] AUROC: 0.8494 / AUPRC: 0.1992
            Trial 2 - Epoch 28/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.03it/s]
            [Train] AUROC: 0.9486 / AUPRC: 0.5144
            [Valid] AUROC: 0.8637 / AUPRC: 0.2156
            Epoch 00028: reducing learning rate of group 0 to 6.6744e-05.
            Trial 2 - Epoch 29/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.22it/s]
            [Train] AUROC: 0.9536 / AUPRC: 0.5314
            [Valid] AUROC: 0.9141 / AUPRC: 0.3038
            Trial 2 - Epoch 30/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.33it/s]
            [Train] AUROC: 0.9557 / AUPRC: 0.5538
            [Valid] AUROC: 0.9080 / AUPRC: 0.2895
            Trial 2 - Epoch 31/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.35it/s]
            [Train] AUROC: 0.9545 / AUPRC: 0.5593
            [Valid] AUROC: 0.9043 / AUPRC: 0.2815
            Trial 2 - Epoch 32/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.24it/s]
            [Train] AUROC: 0.9552 / AUPRC: 0.5457
            [Valid] AUROC: 0.9085 / AUPRC: 0.2906
            Trial 2 - Epoch 33/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.54it/s]
            [Train] AUROC: 0.9563 / AUPRC: 0.5729
            [Valid] AUROC: 0.9036 / AUPRC: 0.2800
            Trial 2 - Epoch 34/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.56it/s]
            [Train] AUROC: 0.9570 / AUPRC: 0.5646
            [Valid] AUROC: 0.9112 / AUPRC: 0.2969
            Trial 2 - Epoch 35/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.03it/s]
            [Train] AUROC: 0.9588 / AUPRC: 0.5714
            [Valid] AUROC: 0.8809 / AUPRC: 0.2394
            Epoch 00035: reducing learning rate of group 0 to 6.6744e-06.
            Trial 2 - Epoch 36/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.26it/s]
            [Train] AUROC: 0.9575 / AUPRC: 0.5629
            [Valid] AUROC: 0.9162 / AUPRC: 0.3092
            Trial 2 - Epoch 37/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.14it/s]
            [Train] AUROC: 0.9573 / AUPRC: 0.5523
            [Valid] AUROC: 0.9225 / AUPRC: 0.3265
            Trial 2 - Epoch 38/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.58it/s]
            [Train] AUROC: 0.9559 / AUPRC: 0.5578
            [Valid] AUROC: 0.9094 / AUPRC: 0.2926
            Trial 2 - Epoch 39/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.41it/s]
            [Train] AUROC: 0.9580 / AUPRC: 0.5644
            [Valid] AUROC: 0.9113 / AUPRC: 0.2972
            Trial 2 - Epoch 40/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.05it/s]
            [Train] AUROC: 0.9603 / AUPRC: 0.5882
            [Valid] AUROC: 0.9095 / AUPRC: 0.2929
            Trial 2 - Epoch 41/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.35it/s]
            [Train] AUROC: 0.9562 / AUPRC: 0.5570
            [Valid] AUROC: 0.9133 / AUPRC: 0.3019
            Trial 2 - Epoch 42/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.21it/s]
            [Train] AUROC: 0.9584 / AUPRC: 0.5663
            [Valid] AUROC: 0.9156 / AUPRC: 0.3075
            Trial 2 - Epoch 43/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.21it/s]
            [Train] AUROC: 0.9573 / AUPRC: 0.5560
            [Valid] AUROC: 0.8479 / AUPRC: 0.1977
            Epoch 00043: reducing learning rate of group 0 to 6.6744e-07.
            Trial 2 - Epoch 44/44 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.26it/s]
            [Train] AUROC: 0.9588 / AUPRC: 0.5623
            [I 2023-12-12 15:44:27,150] Trial 1 finished with value: 0.32647759694367967 and parameters: {'learning_rate': 0.0006674418213852289, 'batch_size': 32, 'num_epochs': 44, 'optimizer': 'SGD'}. Best is trial 1 with value: 0.32647759694367967.
            [Valid] AUROC: 0.9057 / AUPRC: 0.2844
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.1326
            [valid_loss]: 0.2178
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.9573
            [valid_auroc]: 0.9225
            [train_auprc]: 0.5523
            [valid_auprc]: 0.3265 <- Pick It Up!
            
            ## [Learning Rate] : 0.00010021334889412966
            [Batch Size] : 64
            [Epochs] : 35
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.00010021334889412966
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 3 - Epoch 1/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.37it/s]
            
            > [Train] AUROC: 0.5010 / AUPRC: 0.0685
            [Valid] AUROC: 0.4662 / AUPRC: 0.0663
            Trial 3 - Epoch 2/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.03it/s]
            [Train] AUROC: 0.5060 / AUPRC: 0.0699
            [Valid] AUROC: 0.4470 / AUPRC: 0.0626
            Trial 3 - Epoch 3/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.98it/s]
            [Train] AUROC: 0.5025 / AUPRC: 0.0719
            [Valid] AUROC: 0.4806 / AUPRC: 0.0674
            Trial 3 - Epoch 4/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.07it/s]
            [Train] AUROC: 0.4905 / AUPRC: 0.0690
            [Valid] AUROC: 0.4787 / AUPRC: 0.0693
            Trial 3 - Epoch 5/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.43it/s]
            [Train] AUROC: 0.4966 / AUPRC: 0.0682
            [Valid] AUROC: 0.5046 / AUPRC: 0.0783
            Trial 3 - Epoch 6/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.86it/s]
            [Train] AUROC: 0.5026 / AUPRC: 0.0705
            [Valid] AUROC: 0.4729 / AUPRC: 0.0684
            Trial 3 - Epoch 7/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.78it/s]
            [Train] AUROC: 0.4946 / AUPRC: 0.0692
            [Valid] AUROC: 0.4738 / AUPRC: 0.0693
            Trial 3 - Epoch 8/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.02it/s]
            [Train] AUROC: 0.5140 / AUPRC: 0.0713
            [Valid] AUROC: 0.5426 / AUPRC: 0.0779
            Trial 3 - Epoch 9/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.06it/s]
            [Train] AUROC: 0.5012 / AUPRC: 0.0690
            [Valid] AUROC: 0.5327 / AUPRC: 0.0845
            Trial 3 - Epoch 10/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.09it/s]
            [Train] AUROC: 0.4857 / AUPRC: 0.0675
            [Valid] AUROC: 0.5535 / AUPRC: 0.0932
            Trial 3 - Epoch 11/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.87it/s]
            [Train] AUROC: 0.5071 / AUPRC: 0.0708
            [Valid] AUROC: 0.5289 / AUPRC: 0.0855
            Trial 3 - Epoch 12/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.96it/s]
            [Train] AUROC: 0.4931 / AUPRC: 0.0686
            [Valid] AUROC: 0.5984 / AUPRC: 0.1040
            Trial 3 - Epoch 13/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.96it/s]
            [Train] AUROC: 0.5016 / AUPRC: 0.0689
            [Valid] AUROC: 0.5823 / AUPRC: 0.0985
            Trial 3 - Epoch 14/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.30it/s]
            [Train] AUROC: 0.4993 / AUPRC: 0.0683
            [Valid] AUROC: 0.6172 / AUPRC: 0.1095
            Trial 3 - Epoch 15/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.47it/s]
            [Train] AUROC: 0.5097 / AUPRC: 0.0742
            [Valid] AUROC: 0.5604 / AUPRC: 0.0896
            Trial 3 - Epoch 16/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.92it/s]
            [Train] AUROC: 0.5015 / AUPRC: 0.0697
            [Valid] AUROC: 0.5705 / AUPRC: 0.0948
            Trial 3 - Epoch 17/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.36it/s]
            [Train] AUROC: 0.5135 / AUPRC: 0.0737
            [Valid] AUROC: 0.5801 / AUPRC: 0.0962
            Trial 3 - Epoch 18/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.24it/s]
            [Train] AUROC: 0.4983 / AUPRC: 0.0693
            [Valid] AUROC: 0.5944 / AUPRC: 0.0972
            Trial 3 - Epoch 19/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.28it/s]
            [Train] AUROC: 0.5065 / AUPRC: 0.0702
            [Valid] AUROC: 0.5669 / AUPRC: 0.0981
            Trial 3 - Epoch 20/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.43it/s]
            [Train] AUROC: 0.5054 / AUPRC: 0.0715
            [Valid] AUROC: 0.5919 / AUPRC: 0.1026
            Epoch 00020: reducing learning rate of group 0 to 1.0021e-05.
            Trial 3 - Epoch 21/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.31it/s]
            [Train] AUROC: 0.5109 / AUPRC: 0.0716
            [Valid] AUROC: 0.5901 / AUPRC: 0.1046
            Trial 3 - Epoch 22/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.56it/s]
            [Train] AUROC: 0.5002 / AUPRC: 0.0691
            [Valid] AUROC: 0.5907 / AUPRC: 0.1003
            Trial 3 - Epoch 23/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.58it/s]
            [Train] AUROC: 0.5073 / AUPRC: 0.0706
            [Valid] AUROC: 0.5570 / AUPRC: 0.0915
            Trial 3 - Epoch 24/35 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.86it/s]
            [Train] AUROC: 0.5148 / AUPRC: 0.0738
            [I 2023-12-12 15:47:38,943] Trial 2 finished with value: 0.10951341461546527 and parameters: {'learning_rate': 0.00010021334889412966, 'batch_size': 64, 'num_epochs': 35, 'optimizer': 'SGD'}. Best is trial 1 with value: 0.32647759694367967.
            [Valid] AUROC: 0.5811 / AUPRC: 0.0949
            Early stopping
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.2664
            [valid_loss]: 0.3163
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.4993
            [valid_auroc]: 0.6172
            [train_auprc]: 0.0683
            [valid_auprc]: 0.1095 <- Pick It Up!
            
            ## [Learning Rate] : 0.0001510230051340678
            [Batch Size] : 128
            [Epochs] : 37
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.0001510230051340678
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 4 - Epoch 1/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.60it/s]
            
            > [Train] AUROC: 0.4943 / AUPRC: 0.0683
            [Valid] AUROC: 0.4821 / AUPRC: 0.0673
            Trial 4 - Epoch 2/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.08it/s]
            [Train] AUROC: 0.5021 / AUPRC: 0.0701
            [Valid] AUROC: 0.4727 / AUPRC: 0.0661
            Trial 4 - Epoch 3/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.11it/s]
            [Train] AUROC: 0.4877 / AUPRC: 0.0666
            [Valid] AUROC: 0.4839 / AUPRC: 0.0689
            Trial 4 - Epoch 4/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.07it/s]
            [Train] AUROC: 0.5031 / AUPRC: 0.0699
            [Valid] AUROC: 0.4648 / AUPRC: 0.0653
            Trial 4 - Epoch 5/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.07it/s]
            [Train] AUROC: 0.5059 / AUPRC: 0.0705
            [Valid] AUROC: 0.5127 / AUPRC: 0.0751
            Trial 4 - Epoch 6/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.48it/s]
            [Train] AUROC: 0.5097 / AUPRC: 0.0708
            [Valid] AUROC: 0.5229 / AUPRC: 0.0791
            Trial 4 - Epoch 7/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.62it/s]
            [Train] AUROC: 0.4827 / AUPRC: 0.0661
            [Valid] AUROC: 0.5379 / AUPRC: 0.0823
            Trial 4 - Epoch 8/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.60it/s]
            [Train] AUROC: 0.5049 / AUPRC: 0.0715
            [Valid] AUROC: 0.4709 / AUPRC: 0.0643
            Trial 4 - Epoch 9/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.54it/s]
            [Train] AUROC: 0.4890 / AUPRC: 0.0686
            [Valid] AUROC: 0.5174 / AUPRC: 0.0739
            Trial 4 - Epoch 10/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.75it/s]
            [Train] AUROC: 0.5058 / AUPRC: 0.0705
            [Valid] AUROC: 0.5217 / AUPRC: 0.0761
            Trial 4 - Epoch 11/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.39it/s]
            [Train] AUROC: 0.5075 / AUPRC: 0.0714
            [Valid] AUROC: 0.4739 / AUPRC: 0.0663
            Trial 4 - Epoch 12/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.39it/s]
            [Train] AUROC: 0.5086 / AUPRC: 0.0716
            [Valid] AUROC: 0.5160 / AUPRC: 0.0748
            Trial 4 - Epoch 13/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.14it/s]
            [Train] AUROC: 0.4950 / AUPRC: 0.0694
            [Valid] AUROC: 0.5350 / AUPRC: 0.0779
            Epoch 00013: reducing learning rate of group 0 to 1.5102e-05.
            Trial 4 - Epoch 14/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.48it/s]
            [Train] AUROC: 0.5082 / AUPRC: 0.0714
            [Valid] AUROC: 0.5470 / AUPRC: 0.0847
            Trial 4 - Epoch 15/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.77it/s]
            [Train] AUROC: 0.5075 / AUPRC: 0.0703
            [Valid] AUROC: 0.5243 / AUPRC: 0.0759
            Trial 4 - Epoch 16/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.09it/s]
            [Train] AUROC: 0.4890 / AUPRC: 0.0675
            [Valid] AUROC: 0.5251 / AUPRC: 0.0760
            Trial 4 - Epoch 17/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.36it/s]
            [Train] AUROC: 0.5120 / AUPRC: 0.0734
            [Valid] AUROC: 0.5265 / AUPRC: 0.0766
            Trial 4 - Epoch 18/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.68it/s]
            [Train] AUROC: 0.4930 / AUPRC: 0.0675
            [Valid] AUROC: 0.5394 / AUPRC: 0.0796
            Trial 4 - Epoch 19/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.62it/s]
            [Train] AUROC: 0.5191 / AUPRC: 0.0744
            [Valid] AUROC: 0.5555 / AUPRC: 0.0848
            Trial 4 - Epoch 20/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.33it/s]
            [Train] AUROC: 0.5028 / AUPRC: 0.0694
            [Valid] AUROC: 0.5388 / AUPRC: 0.0790
            Trial 4 - Epoch 21/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.49it/s]
            [Train] AUROC: 0.5044 / AUPRC: 0.0694
            [Valid] AUROC: 0.5429 / AUPRC: 0.0826
            Trial 4 - Epoch 22/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.76it/s]
            [Train] AUROC: 0.5115 / AUPRC: 0.0724
            [Valid] AUROC: 0.5420 / AUPRC: 0.0825
            Trial 4 - Epoch 23/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.19it/s]
            [Train] AUROC: 0.4995 / AUPRC: 0.0702
            [Valid] AUROC: 0.5180 / AUPRC: 0.0752
            Trial 4 - Epoch 24/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.56it/s]
            [Train] AUROC: 0.5130 / AUPRC: 0.0738
            [Valid] AUROC: 0.5680 / AUPRC: 0.0887
            Trial 4 - Epoch 25/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.54it/s]
            [Train] AUROC: 0.5137 / AUPRC: 0.0740
            [Valid] AUROC: 0.4917 / AUPRC: 0.0672
            Trial 4 - Epoch 26/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.57it/s]
            [Train] AUROC: 0.5038 / AUPRC: 0.0697
            [Valid] AUROC: 0.5182 / AUPRC: 0.0732
            Trial 4 - Epoch 27/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.71it/s]
            [Train] AUROC: 0.5042 / AUPRC: 0.0699
            [Valid] AUROC: 0.5385 / AUPRC: 0.0763
            Trial 4 - Epoch 28/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.63it/s]
            [Train] AUROC: 0.4959 / AUPRC: 0.0695
            [Valid] AUROC: 0.5538 / AUPRC: 0.0833
            Trial 4 - Epoch 29/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.55it/s]
            [Train] AUROC: 0.5062 / AUPRC: 0.0719
            [Valid] AUROC: 0.5478 / AUPRC: 0.0826
            Trial 4 - Epoch 30/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.57it/s]
            [Train] AUROC: 0.4954 / AUPRC: 0.0682
            [Valid] AUROC: 0.5585 / AUPRC: 0.0840
            Epoch 00030: reducing learning rate of group 0 to 1.5102e-06.
            Trial 4 - Epoch 31/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.54it/s]
            [Train] AUROC: 0.5079 / AUPRC: 0.0703
            [Valid] AUROC: 0.5571 / AUPRC: 0.0853
            Trial 4 - Epoch 32/37 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.58it/s]
            [Train] AUROC: 0.4889 / AUPRC: 0.0672
            [Valid] AUROC: 0.5274 / AUPRC: 0.0780
            Trial 4 - Epoch 33/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.54it/s]
            [Train] AUROC: 0.4975 / AUPRC: 0.0677
            [Valid] AUROC: 0.5297 / AUPRC: 0.0772
            Trial 4 - Epoch 34/37 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.52it/s]
            [Train] AUROC: 0.4946 / AUPRC: 0.0676
            [I 2023-12-12 15:51:53,633] Trial 3 finished with value: 0.08874518974090129 and parameters: {'learning_rate': 0.0001510230051340678, 'batch_size': 128, 'num_epochs': 37, 'optimizer': 'SGD'}. Best is trial 1 with value: 0.32647759694367967.
            [Valid] AUROC: 0.5385 / AUPRC: 0.0809
            Early stopping
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.2696
            [valid_loss]: 0.3282
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.5130
            [valid_auroc]: 0.5680
            [train_auprc]: 0.0738
            [valid_auprc]: 0.0887 <- Pick It Up!
            
            ## [Learning Rate] : 0.0004178442199935877
            [Batch Size] : 64
            [Epochs] : 36
            [Optimizer] : Adam (
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.0004178442199935877
            maximize: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 5 - Epoch 1/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.17it/s]
            
            > [Train] AUROC: 0.4988 / AUPRC: 0.0693
            [Valid] AUROC: 0.5843 / AUPRC: 0.0889
            Trial 5 - Epoch 2/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.52it/s]
            [Train] AUROC: 0.5168 / AUPRC: 0.0736
            [Valid] AUROC: 0.3095 / AUPRC: 0.0470
            Trial 5 - Epoch 3/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.43it/s]
            [Train] AUROC: 0.6434 / AUPRC: 0.1209
            [Valid] AUROC: 0.5287 / AUPRC: 0.2496
            Trial 5 - Epoch 4/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.43it/s]
            [Train] AUROC: 0.9177 / AUPRC: 0.4750
            [Valid] AUROC: 0.9248 / AUPRC: 0.5615
            Trial 5 - Epoch 5/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.64it/s]
            [Train] AUROC: 0.9593 / AUPRC: 0.6232
            [Valid] AUROC: 0.9620 / AUPRC: 0.6426
            Trial 5 - Epoch 6/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.77it/s]
            [Train] AUROC: 0.9685 / AUPRC: 0.6490
            [Valid] AUROC: 0.9805 / AUPRC: 0.6844
            Trial 5 - Epoch 7/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.26it/s]
            [Train] AUROC: 0.9720 / AUPRC: 0.6840
            [Valid] AUROC: 0.9837 / AUPRC: 0.7170
            Trial 5 - Epoch 8/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.16it/s]
            [Train] AUROC: 0.9748 / AUPRC: 0.7076
            [Valid] AUROC: 0.9826 / AUPRC: 0.6822
            Trial 5 - Epoch 9/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.01it/s]
            [Train] AUROC: 0.9753 / AUPRC: 0.7531
            [Valid] AUROC: 0.9847 / AUPRC: 0.7692
            Trial 5 - Epoch 10/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.62it/s]
            [Train] AUROC: 0.9794 / AUPRC: 0.7755
            [Valid] AUROC: 0.9868 / AUPRC: 0.7680
            Trial 5 - Epoch 11/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.54it/s]
            [Train] AUROC: 0.9810 / AUPRC: 0.7679
            [Valid] AUROC: 0.9893 / AUPRC: 0.8376
            Trial 5 - Epoch 12/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.72it/s]
            [Train] AUROC: 0.9812 / AUPRC: 0.7744
            [Valid] AUROC: 0.9871 / AUPRC: 0.8100
            Trial 5 - Epoch 13/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.00it/s]
            [Train] AUROC: 0.9830 / AUPRC: 0.8026
            [Valid] AUROC: 0.9901 / AUPRC: 0.8353
            Trial 5 - Epoch 14/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.69it/s]
            [Train] AUROC: 0.9843 / AUPRC: 0.7966
            [Valid] AUROC: 0.9846 / AUPRC: 0.7579
            Trial 5 - Epoch 15/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.46it/s]
            [Train] AUROC: 0.9853 / AUPRC: 0.8444
            [Valid] AUROC: 0.9898 / AUPRC: 0.8081
            Trial 5 - Epoch 16/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.73it/s]
            [Train] AUROC: 0.9860 / AUPRC: 0.8427
            [Valid] AUROC: 0.9886 / AUPRC: 0.8207
            Trial 5 - Epoch 17/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.9862 / AUPRC: 0.8300
            [Valid] AUROC: 0.9898 / AUPRC: 0.8843
            Trial 5 - Epoch 18/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.31it/s]
            [Train] AUROC: 0.9857 / AUPRC: 0.8473
            [Valid] AUROC: 0.9889 / AUPRC: 0.8459
            Trial 5 - Epoch 19/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.70it/s]
            [Train] AUROC: 0.9877 / AUPRC: 0.8606
            [Valid] AUROC: 0.9907 / AUPRC: 0.9036
            Trial 5 - Epoch 20/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.81it/s]
            [Train] AUROC: 0.9884 / AUPRC: 0.8759
            [Valid] AUROC: 0.9874 / AUPRC: 0.9008
            Trial 5 - Epoch 21/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.75it/s]
            [Train] AUROC: 0.9901 / AUPRC: 0.8897
            [Valid] AUROC: 0.9897 / AUPRC: 0.8632
            Trial 5 - Epoch 22/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.74it/s]
            [Train] AUROC: 0.9920 / AUPRC: 0.9051
            [Valid] AUROC: 0.9886 / AUPRC: 0.8315
            Trial 5 - Epoch 23/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.25it/s]
            [Train] AUROC: 0.9922 / AUPRC: 0.9122
            [Valid] AUROC: 0.9865 / AUPRC: 0.9076
            Trial 5 - Epoch 24/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.66it/s]
            [Train] AUROC: 0.9932 / AUPRC: 0.9253
            [Valid] AUROC: 0.9912 / AUPRC: 0.8994
            Trial 5 - Epoch 25/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.15it/s]
            [Train] AUROC: 0.9933 / AUPRC: 0.9286
            [Valid] AUROC: 0.9887 / AUPRC: 0.8903
            Trial 5 - Epoch 26/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.16it/s]
            [Train] AUROC: 0.9926 / AUPRC: 0.9323
            [Valid] AUROC: 0.9919 / AUPRC: 0.9053
            Trial 5 - Epoch 27/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.09it/s]
            [Train] AUROC: 0.9937 / AUPRC: 0.9428
            [Valid] AUROC: 0.9901 / AUPRC: 0.8903
            Trial 5 - Epoch 28/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.20it/s]
            [Train] AUROC: 0.9922 / AUPRC: 0.9383
            [Valid] AUROC: 0.9909 / AUPRC: 0.9169
            Trial 5 - Epoch 29/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.28it/s]
            [Train] AUROC: 0.9929 / AUPRC: 0.9456
            [Valid] AUROC: 0.9876 / AUPRC: 0.9146
            Trial 5 - Epoch 30/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.39it/s]
            [Train] AUROC: 0.9942 / AUPRC: 0.9485
            [Valid] AUROC: 0.9902 / AUPRC: 0.8998
            Trial 5 - Epoch 31/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.34it/s]
            [Train] AUROC: 0.9945 / AUPRC: 0.9550
            [Valid] AUROC: 0.9906 / AUPRC: 0.9011
            Trial 5 - Epoch 32/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.67it/s]
            [Train] AUROC: 0.9968 / AUPRC: 0.9672
            [Valid] AUROC: 0.9851 / AUPRC: 0.8942
            Trial 5 - Epoch 33/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.56it/s]
            [Train] AUROC: 0.9958 / AUPRC: 0.9657
            [Valid] AUROC: 0.9877 / AUPRC: 0.8969
            Trial 5 - Epoch 34/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.50it/s]
            [Train] AUROC: 0.9948 / AUPRC: 0.9559
            [Valid] AUROC: 0.9913 / AUPRC: 0.9133
            Epoch 00034: reducing learning rate of group 0 to 4.1784e-05.
            Trial 5 - Epoch 35/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.45it/s]
            [Train] AUROC: 0.9979 / AUPRC: 0.9771
            [Valid] AUROC: 0.9901 / AUPRC: 0.9178
            Trial 5 - Epoch 36/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.62it/s]
            [Train] AUROC: 0.9980 / AUPRC: 0.9856
            [I 2023-12-12 15:56:40,699] Trial 4 finished with value: 0.9213366982167329 and parameters: {'learning_rate': 0.0004178442199935877, 'batch_size': 64, 'num_epochs': 36, 'optimizer': 'Adam'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.9914 / AUPRC: 0.9213
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.0198
            [valid_loss]: 0.0727
            [train_accuracy]: 0.9951
            [valid_accuracy]: 0.9761
            [train_auroc]: 0.9980
            [valid_auroc]: 0.9914
            [train_auprc]: 0.9856
            [valid_auprc]: 0.9213 <- Pick It Up!
            
            ## [Learning Rate] : 0.00014094265233619555
            [Batch Size] : 64
            [Epochs] : 45
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.00014094265233619555
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 6 - Epoch 1/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.86it/s]
            
            > [Train] AUROC: 0.5015 / AUPRC: 0.0686
            [Valid] AUROC: 0.4720 / AUPRC: 0.0673
            Trial 6 - Epoch 2/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.81it/s]
            [Train] AUROC: 0.5055 / AUPRC: 0.0697
            [Valid] AUROC: 0.4596 / AUPRC: 0.0650
            Trial 6 - Epoch 3/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.51it/s]
            [Train] AUROC: 0.5025 / AUPRC: 0.0712
            [Valid] AUROC: 0.4728 / AUPRC: 0.0651
            Trial 6 - Epoch 4/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.74it/s]
            [Train] AUROC: 0.4928 / AUPRC: 0.0710
            [Valid] AUROC: 0.4632 / AUPRC: 0.0653
            Trial 6 - Epoch 5/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.25it/s]
            [Train] AUROC: 0.4970 / AUPRC: 0.0684
            [Valid] AUROC: 0.4983 / AUPRC: 0.0742
            Trial 6 - Epoch 6/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.56it/s]
            [Train] AUROC: 0.5037 / AUPRC: 0.0705
            [Valid] AUROC: 0.5670 / AUPRC: 0.0972
            Trial 6 - Epoch 7/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.84it/s]
            [Train] AUROC: 0.4941 / AUPRC: 0.0696
            [Valid] AUROC: 0.5311 / AUPRC: 0.0810
            Trial 6 - Epoch 8/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.07it/s]
            [Train] AUROC: 0.5130 / AUPRC: 0.0708
            [Valid] AUROC: 0.5167 / AUPRC: 0.0752
            Trial 6 - Epoch 9/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.88it/s]
            [Train] AUROC: 0.4994 / AUPRC: 0.0684
            [Valid] AUROC: 0.5746 / AUPRC: 0.0945
            Trial 6 - Epoch 10/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.72it/s]
            [Train] AUROC: 0.4885 / AUPRC: 0.0676
            [Valid] AUROC: 0.6079 / AUPRC: 0.1073
            Trial 6 - Epoch 11/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.90it/s]
            [Train] AUROC: 0.5052 / AUPRC: 0.0705
            [Valid] AUROC: 0.5629 / AUPRC: 0.0908
            Trial 6 - Epoch 12/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.33it/s]
            [Train] AUROC: 0.4933 / AUPRC: 0.0687
            [Valid] AUROC: 0.5553 / AUPRC: 0.0938
            Trial 6 - Epoch 13/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.46it/s]
            [Train] AUROC: 0.4999 / AUPRC: 0.0686
            [Valid] AUROC: 0.5665 / AUPRC: 0.0900
            Trial 6 - Epoch 14/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.20it/s]
            [Train] AUROC: 0.4997 / AUPRC: 0.0677
            [Valid] AUROC: 0.5828 / AUPRC: 0.0853
            Trial 6 - Epoch 15/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.42it/s]
            [Train] AUROC: 0.5123 / AUPRC: 0.0730
            [Valid] AUROC: 0.5978 / AUPRC: 0.1031
            Trial 6 - Epoch 16/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.36it/s]
            [Train] AUROC: 0.4991 / AUPRC: 0.0701
            [Valid] AUROC: 0.6138 / AUPRC: 0.1047
            Epoch 00016: reducing learning rate of group 0 to 1.4094e-05.
            Trial 6 - Epoch 17/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.80it/s]
            [Train] AUROC: 0.5136 / AUPRC: 0.0712
            [Valid] AUROC: 0.6182 / AUPRC: 0.1130
            Trial 6 - Epoch 18/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.83it/s]
            [Train] AUROC: 0.4986 / AUPRC: 0.0687
            [Valid] AUROC: 0.6116 / AUPRC: 0.1114
            Trial 6 - Epoch 19/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.17it/s]
            [Train] AUROC: 0.5115 / AUPRC: 0.0723
            [Valid] AUROC: 0.6088 / AUPRC: 0.1161
            Trial 6 - Epoch 20/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.22it/s]
            [Train] AUROC: 0.5069 / AUPRC: 0.0734
            [Valid] AUROC: 0.6135 / AUPRC: 0.1205
            Trial 6 - Epoch 21/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.65it/s]
            [Train] AUROC: 0.5121 / AUPRC: 0.0719
            [Valid] AUROC: 0.6192 / AUPRC: 0.1211
            Trial 6 - Epoch 22/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.30it/s]
            [Train] AUROC: 0.5054 / AUPRC: 0.0699
            [Valid] AUROC: 0.5964 / AUPRC: 0.1036
            Trial 6 - Epoch 23/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.06it/s]
            [Train] AUROC: 0.5070 / AUPRC: 0.0709
            [Valid] AUROC: 0.6193 / AUPRC: 0.1158
            Trial 6 - Epoch 24/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.05it/s]
            [Train] AUROC: 0.5149 / AUPRC: 0.0728
            [Valid] AUROC: 0.6331 / AUPRC: 0.1173
            Trial 6 - Epoch 25/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.90it/s]
            [Train] AUROC: 0.5022 / AUPRC: 0.0713
            [Valid] AUROC: 0.5997 / AUPRC: 0.0941
            Trial 6 - Epoch 26/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.64it/s]
            [Train] AUROC: 0.5193 / AUPRC: 0.0728
            [Valid] AUROC: 0.6077 / AUPRC: 0.0996
            Trial 6 - Epoch 27/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 37.00it/s]
            [Train] AUROC: 0.4871 / AUPRC: 0.0671
            [Valid] AUROC: 0.5776 / AUPRC: 0.0906
            Epoch 00027: reducing learning rate of group 0 to 1.4094e-06.
            Trial 6 - Epoch 28/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.44it/s]
            [Train] AUROC: 0.5034 / AUPRC: 0.0685
            [Valid] AUROC: 0.6258 / AUPRC: 0.1234
            Trial 6 - Epoch 29/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.00it/s]
            [Train] AUROC: 0.5116 / AUPRC: 0.0725
            [Valid] AUROC: 0.6216 / AUPRC: 0.1145
            Trial 6 - Epoch 30/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.78it/s]
            [Train] AUROC: 0.5094 / AUPRC: 0.0710
            [Valid] AUROC: 0.6301 / AUPRC: 0.1298
            Trial 6 - Epoch 31/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.66it/s]
            [Train] AUROC: 0.5216 / AUPRC: 0.0771
            [Valid] AUROC: 0.6274 / AUPRC: 0.1324
            Trial 6 - Epoch 32/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.81it/s]
            [Train] AUROC: 0.5122 / AUPRC: 0.0705
            [Valid] AUROC: 0.6248 / AUPRC: 0.1132
            Trial 6 - Epoch 33/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.69it/s]
            [Train] AUROC: 0.5082 / AUPRC: 0.0707
            [Valid] AUROC: 0.6240 / AUPRC: 0.1191
            Trial 6 - Epoch 34/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.51it/s]
            [Train] AUROC: 0.5054 / AUPRC: 0.0702
            [Valid] AUROC: 0.6357 / AUPRC: 0.1261
            Trial 6 - Epoch 35/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.12it/s]
            [Train] AUROC: 0.5020 / AUPRC: 0.0699
            [Valid] AUROC: 0.6354 / AUPRC: 0.1257
            Trial 6 - Epoch 36/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.41it/s]
            [Train] AUROC: 0.4869 / AUPRC: 0.0681
            [Valid] AUROC: 0.6300 / AUPRC: 0.1172
            Trial 6 - Epoch 37/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.35it/s]
            [Train] AUROC: 0.5100 / AUPRC: 0.0708
            [Valid] AUROC: 0.6301 / AUPRC: 0.1293
            Epoch 00037: reducing learning rate of group 0 to 1.4094e-07.
            Trial 6 - Epoch 38/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.48it/s]
            [Train] AUROC: 0.5056 / AUPRC: 0.0707
            [Valid] AUROC: 0.6237 / AUPRC: 0.1172
            Trial 6 - Epoch 39/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.16it/s]
            [Train] AUROC: 0.5055 / AUPRC: 0.0728
            [Valid] AUROC: 0.6256 / AUPRC: 0.1206
            Trial 6 - Epoch 40/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.20it/s]
            [Train] AUROC: 0.5166 / AUPRC: 0.0719
            [Valid] AUROC: 0.6347 / AUPRC: 0.1246
            Trial 6 - Epoch 41/45 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.28it/s]
            [Train] AUROC: 0.5104 / AUPRC: 0.0716
            [I 2023-12-12 16:02:07,836] Trial 5 finished with value: 0.1324467007852967 and parameters: {'learning_rate': 0.00014094265233619555, 'batch_size': 64, 'num_epochs': 45, 'optimizer': 'SGD'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.6090 / AUPRC: 0.1047
            Early stopping
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.2584
            [valid_loss]: 0.3078
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.5216
            [valid_auroc]: 0.6274
            [train_auprc]: 0.0771
            [valid_auprc]: 0.1324 <- Pick It Up!
            
            ## [Learning Rate] : 1.3215394746016554e-05
            [Batch Size] : 32
            [Epochs] : 30
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 1.3215394746016554e-05
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 7 - Epoch 1/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.40it/s]
            
            > [Train] AUROC: 0.5095 / AUPRC: 0.0715
            [Valid] AUROC: 0.4310 / AUPRC: 0.0614
            Trial 7 - Epoch 2/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.56it/s]
            [Train] AUROC: 0.4968 / AUPRC: 0.0690
            [Valid] AUROC: 0.4755 / AUPRC: 0.0669
            Trial 7 - Epoch 3/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.09it/s]
            [Train] AUROC: 0.4805 / AUPRC: 0.0650
            [Valid] AUROC: 0.4645 / AUPRC: 0.0654
            Trial 7 - Epoch 4/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.11it/s]
            [Train] AUROC: 0.4980 / AUPRC: 0.0704
            [Valid] AUROC: 0.4787 / AUPRC: 0.0686
            Trial 7 - Epoch 5/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.58it/s]
            [Train] AUROC: 0.4844 / AUPRC: 0.0664
            [Valid] AUROC: 0.4528 / AUPRC: 0.0633
            Trial 7 - Epoch 6/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.19it/s]
            [Train] AUROC: 0.5031 / AUPRC: 0.0709
            [Valid] AUROC: 0.5020 / AUPRC: 0.0743
            Trial 7 - Epoch 7/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 61.55it/s]
            [Train] AUROC: 0.4972 / AUPRC: 0.0714
            [Valid] AUROC: 0.5073 / AUPRC: 0.0743
            Trial 7 - Epoch 8/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.01it/s]
            [Train] AUROC: 0.5061 / AUPRC: 0.0699
            [Valid] AUROC: 0.5036 / AUPRC: 0.0727
            Trial 7 - Epoch 9/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.02it/s]
            [Train] AUROC: 0.4936 / AUPRC: 0.0675
            [Valid] AUROC: 0.5166 / AUPRC: 0.0784
            Trial 7 - Epoch 10/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.09it/s]
            [Train] AUROC: 0.5133 / AUPRC: 0.0718
            [Valid] AUROC: 0.4824 / AUPRC: 0.0721
            Trial 7 - Epoch 11/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.73it/s]
            [Train] AUROC: 0.4988 / AUPRC: 0.0699
            [Valid] AUROC: 0.5124 / AUPRC: 0.0780
            Trial 7 - Epoch 12/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.50it/s]
            [Train] AUROC: 0.4958 / AUPRC: 0.0720
            [Valid] AUROC: 0.5399 / AUPRC: 0.0832
            Trial 7 - Epoch 13/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.35it/s]
            [Train] AUROC: 0.5110 / AUPRC: 0.0713
            [Valid] AUROC: 0.5252 / AUPRC: 0.0793
            Trial 7 - Epoch 14/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.51it/s]
            [Train] AUROC: 0.5087 / AUPRC: 0.0700
            [Valid] AUROC: 0.5154 / AUPRC: 0.0756
            Trial 7 - Epoch 15/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.21it/s]
            [Train] AUROC: 0.4944 / AUPRC: 0.0680
            [Valid] AUROC: 0.5227 / AUPRC: 0.0785
            Trial 7 - Epoch 16/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.58it/s]
            [Train] AUROC: 0.5011 / AUPRC: 0.0703
            [Valid] AUROC: 0.5056 / AUPRC: 0.0770
            Trial 7 - Epoch 17/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.49it/s]
            [Train] AUROC: 0.4893 / AUPRC: 0.0657
            [Valid] AUROC: 0.5431 / AUPRC: 0.0881
            Trial 7 - Epoch 18/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.56it/s]
            [Train] AUROC: 0.4994 / AUPRC: 0.0697
            [Valid] AUROC: 0.5423 / AUPRC: 0.0838
            Trial 7 - Epoch 19/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.88it/s]
            [Train] AUROC: 0.5067 / AUPRC: 0.0714
            [Valid] AUROC: 0.5058 / AUPRC: 0.0766
            Trial 7 - Epoch 20/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.15it/s]
            [Train] AUROC: 0.4993 / AUPRC: 0.0699
            [Valid] AUROC: 0.5177 / AUPRC: 0.0775
            Trial 7 - Epoch 21/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.31it/s]
            [Train] AUROC: 0.5008 / AUPRC: 0.0700
            [Valid] AUROC: 0.5308 / AUPRC: 0.0786
            Trial 7 - Epoch 22/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.56it/s]
            [Train] AUROC: 0.5056 / AUPRC: 0.0708
            [Valid] AUROC: 0.5002 / AUPRC: 0.0749
            Trial 7 - Epoch 23/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.49it/s]
            [Train] AUROC: 0.4951 / AUPRC: 0.0677
            [Valid] AUROC: 0.5617 / AUPRC: 0.0848
            Epoch 00023: reducing learning rate of group 0 to 1.3215e-06.
            Trial 7 - Epoch 24/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.75it/s]
            [Train] AUROC: 0.5075 / AUPRC: 0.0705
            [Valid] AUROC: 0.5137 / AUPRC: 0.0781
            Trial 7 - Epoch 25/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.76it/s]
            [Train] AUROC: 0.5061 / AUPRC: 0.0692
            [Valid] AUROC: 0.5068 / AUPRC: 0.0734
            Trial 7 - Epoch 26/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.59it/s]
            [Train] AUROC: 0.4955 / AUPRC: 0.0679
            [Valid] AUROC: 0.5138 / AUPRC: 0.0766
            Trial 7 - Epoch 27/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.68it/s]
            [Train] AUROC: 0.5022 / AUPRC: 0.0691
            [I 2023-12-12 16:06:11,923] Trial 6 finished with value: 0.08806206872736355 and parameters: {'learning_rate': 1.3215394746016554e-05, 'batch_size': 32, 'num_epochs': 30, 'optimizer': 'SGD'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.5019 / AUPRC: 0.0755
            Early stopping
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.3133
            [valid_loss]: 0.3952
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.4893
            [valid_auroc]: 0.5431
            [train_auprc]: 0.0657
            [valid_auprc]: 0.0881 <- Pick It Up!
            
            ## [Learning Rate] : 3.308466683166244e-05
            [Batch Size] : 32
            [Epochs] : 30
            [Optimizer] : Adam (
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 3.308466683166244e-05
            maximize: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 8 - Epoch 1/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.20it/s]
            
            > [Train] AUROC: 0.5126 / AUPRC: 0.0731
            [Valid] AUROC: 0.5722 / AUPRC: 0.0832
            Trial 8 - Epoch 2/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 61.34it/s]
            [Train] AUROC: 0.5123 / AUPRC: 0.0710
            [Valid] AUROC: 0.4166 / AUPRC: 0.0570
            Trial 8 - Epoch 3/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.58it/s]
            [Train] AUROC: 0.5109 / AUPRC: 0.0720
            [Valid] AUROC: 0.4287 / AUPRC: 0.0607
            Trial 8 - Epoch 4/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.30it/s]
            [Train] AUROC: 0.5239 / AUPRC: 0.0749
            [Valid] AUROC: 0.3875 / AUPRC: 0.0554
            Trial 8 - Epoch 5/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 61.14it/s]
            [Train] AUROC: 0.5351 / AUPRC: 0.0746
            [Valid] AUROC: 0.3296 / AUPRC: 0.0491
            Trial 8 - Epoch 6/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.23it/s]
            [Train] AUROC: 0.6312 / AUPRC: 0.1078
            [Valid] AUROC: 0.4151 / AUPRC: 0.0561
            Trial 8 - Epoch 7/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 60.75it/s]
            [Train] AUROC: 0.7627 / AUPRC: 0.1797
            [Valid] AUROC: 0.8034 / AUPRC: 0.2076
            Trial 8 - Epoch 8/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 61.57it/s]
            [Train] AUROC: 0.8612 / AUPRC: 0.3131
            [Valid] AUROC: 0.9372 / AUPRC: 0.4294
            Trial 8 - Epoch 9/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 61.07it/s]
            [Train] AUROC: 0.9234 / AUPRC: 0.4523
            [Valid] AUROC: 0.9656 / AUPRC: 0.5684
            Trial 8 - Epoch 10/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.99it/s]
            [Train] AUROC: 0.9474 / AUPRC: 0.5303
            [Valid] AUROC: 0.9706 / AUPRC: 0.5959
            Trial 8 - Epoch 11/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.79it/s]
            [Train] AUROC: 0.9538 / AUPRC: 0.5765
            [Valid] AUROC: 0.9728 / AUPRC: 0.5911
            Trial 8 - Epoch 12/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.83it/s]
            [Train] AUROC: 0.9554 / AUPRC: 0.6103
            [Valid] AUROC: 0.9763 / AUPRC: 0.6144
            Trial 8 - Epoch 13/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.69it/s]
            [Train] AUROC: 0.9641 / AUPRC: 0.6214
            [Valid] AUROC: 0.9783 / AUPRC: 0.6391
            Trial 8 - Epoch 14/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.66it/s]
            [Train] AUROC: 0.9679 / AUPRC: 0.6590
            [Valid] AUROC: 0.9748 / AUPRC: 0.6124
            Trial 8 - Epoch 15/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.63it/s]
            [Train] AUROC: 0.9692 / AUPRC: 0.6815
            [Valid] AUROC: 0.9780 / AUPRC: 0.6399
            Trial 8 - Epoch 16/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.89it/s]
            [Train] AUROC: 0.9694 / AUPRC: 0.6771
            [Valid] AUROC: 0.9770 / AUPRC: 0.6528
            Trial 8 - Epoch 17/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.92it/s]
            [Train] AUROC: 0.9696 / AUPRC: 0.6804
            [Valid] AUROC: 0.9824 / AUPRC: 0.6655
            Trial 8 - Epoch 18/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.81it/s]
            [Train] AUROC: 0.9717 / AUPRC: 0.7047
            [Valid] AUROC: 0.9816 / AUPRC: 0.6754
            Trial 8 - Epoch 19/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.09it/s]
            [Train] AUROC: 0.9713 / AUPRC: 0.6977
            [Valid] AUROC: 0.9829 / AUPRC: 0.6768
            Trial 8 - Epoch 20/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.66it/s]
            [Train] AUROC: 0.9707 / AUPRC: 0.7114
            [Valid] AUROC: 0.9822 / AUPRC: 0.6802
            Trial 8 - Epoch 21/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.38it/s]
            [Train] AUROC: 0.9746 / AUPRC: 0.7217
            [Valid] AUROC: 0.9834 / AUPRC: 0.6816
            Trial 8 - Epoch 22/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.15it/s]
            [Train] AUROC: 0.9752 / AUPRC: 0.7267
            [Valid] AUROC: 0.9788 / AUPRC: 0.6785
            Trial 8 - Epoch 23/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.45it/s]
            [Train] AUROC: 0.9748 / AUPRC: 0.7293
            [Valid] AUROC: 0.9841 / AUPRC: 0.7024
            Trial 8 - Epoch 24/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.24it/s]
            [Train] AUROC: 0.9769 / AUPRC: 0.7304
            [Valid] AUROC: 0.9821 / AUPRC: 0.6834
            Trial 8 - Epoch 25/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.74it/s]
            [Train] AUROC: 0.9765 / AUPRC: 0.7353
            [Valid] AUROC: 0.9789 / AUPRC: 0.6550
            Trial 8 - Epoch 26/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.15it/s]
            [Train] AUROC: 0.9789 / AUPRC: 0.7475
            [Valid] AUROC: 0.9781 / AUPRC: 0.6983
            Trial 8 - Epoch 27/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.82it/s]
            [Train] AUROC: 0.9772 / AUPRC: 0.7524
            [Valid] AUROC: 0.9782 / AUPRC: 0.6893
            Trial 8 - Epoch 28/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.85it/s]
            [Train] AUROC: 0.9803 / AUPRC: 0.7549
            [Valid] AUROC: 0.9840 / AUPRC: 0.7149
            Trial 8 - Epoch 29/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.98it/s]
            [Train] AUROC: 0.9789 / AUPRC: 0.7580
            [Valid] AUROC: 0.9820 / AUPRC: 0.7215
            Trial 8 - Epoch 30/30 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.26it/s]
            [Train] AUROC: 0.9816 / AUPRC: 0.7727
            [I 2023-12-12 16:10:49,228] Trial 7 finished with value: 0.7283733208547105 and parameters: {'learning_rate': 3.308466683166244e-05, 'batch_size': 32, 'num_epochs': 30, 'optimizer': 'Adam'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.9852 / AUPRC: 0.7284
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.0746
            [valid_loss]: 0.0877
            [train_accuracy]: 0.9705
            [valid_accuracy]: 0.9596
            [train_auroc]: 0.9816
            [valid_auroc]: 0.9852
            [train_auprc]: 0.7727
            [valid_auprc]: 0.7284 <- Pick It Up!
            
            ## [Learning Rate] : 0.0004637199702843728
            [Batch Size] : 32
            [Epochs] : 47
            [Optimizer] : SGD (
            Parameter Group 0
            dampening: 0
            differentiable: False
            foreach: None
            lr: 0.0004637199702843728
            maximize: False
            momentum: 0.9
            nesterov: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 9 - Epoch 1/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.50it/s]
            
            > [Train] AUROC: 0.5025 / AUPRC: 0.0704
            [Valid] AUROC: 0.4144 / AUPRC: 0.0594
            Trial 9 - Epoch 2/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.26it/s]
            [Train] AUROC: 0.5113 / AUPRC: 0.0698
            [Valid] AUROC: 0.4233 / AUPRC: 0.0592
            Trial 9 - Epoch 3/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.77it/s]
            [Train] AUROC: 0.5038 / AUPRC: 0.0709
            [Valid] AUROC: 0.4478 / AUPRC: 0.0613
            Trial 9 - Epoch 4/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.69it/s]
            [Train] AUROC: 0.5148 / AUPRC: 0.0721
            [Valid] AUROC: 0.6075 / AUPRC: 0.1044
            Trial 9 - Epoch 5/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 62.36it/s]
            [Train] AUROC: 0.4963 / AUPRC: 0.0676
            [Valid] AUROC: 0.6262 / AUPRC: 0.1128
            Trial 9 - Epoch 6/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.50it/s]
            [Train] AUROC: 0.5182 / AUPRC: 0.0738
            [Valid] AUROC: 0.6416 / AUPRC: 0.1172
            Trial 9 - Epoch 7/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.96it/s]
            [Train] AUROC: 0.5184 / AUPRC: 0.0733
            [Valid] AUROC: 0.6468 / AUPRC: 0.1231
            Trial 9 - Epoch 8/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.49it/s]
            [Train] AUROC: 0.5234 / AUPRC: 0.0745
            [Valid] AUROC: 0.6488 / AUPRC: 0.1170
            Trial 9 - Epoch 9/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.58it/s]
            [Train] AUROC: 0.5076 / AUPRC: 0.0712
            [Valid] AUROC: 0.6485 / AUPRC: 0.1198
            Trial 9 - Epoch 10/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.77it/s]
            [Train] AUROC: 0.5166 / AUPRC: 0.0731
            [Valid] AUROC: 0.6579 / AUPRC: 0.1222
            Trial 9 - Epoch 11/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.33it/s]
            [Train] AUROC: 0.5248 / AUPRC: 0.0735
            [Valid] AUROC: 0.6652 / AUPRC: 0.1244
            Trial 9 - Epoch 12/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.80it/s]
            [Train] AUROC: 0.5375 / AUPRC: 0.0752
            [Valid] AUROC: 0.6761 / AUPRC: 0.1297
            Trial 9 - Epoch 13/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.91it/s]
            [Train] AUROC: 0.5352 / AUPRC: 0.0774
            [Valid] AUROC: 0.6870 / AUPRC: 0.1346
            Trial 9 - Epoch 14/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.17it/s]
            [Train] AUROC: 0.5522 / AUPRC: 0.0827
            [Valid] AUROC: 0.7089 / AUPRC: 0.1533
            Trial 9 - Epoch 15/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.14it/s]
            [Train] AUROC: 0.5742 / AUPRC: 0.0855
            [Valid] AUROC: 0.7311 / AUPRC: 0.1761
            Trial 9 - Epoch 16/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.51it/s]
            [Train] AUROC: 0.6131 / AUPRC: 0.0941
            [Valid] AUROC: 0.7601 / AUPRC: 0.1917
            Trial 9 - Epoch 17/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.76it/s]
            [Train] AUROC: 0.6443 / AUPRC: 0.1118
            [Valid] AUROC: 0.7690 / AUPRC: 0.2001
            Trial 9 - Epoch 18/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.56it/s]
            [Train] AUROC: 0.7143 / AUPRC: 0.1431
            [Valid] AUROC: 0.7693 / AUPRC: 0.1844
            Trial 9 - Epoch 19/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.61it/s]
            [Train] AUROC: 0.7366 / AUPRC: 0.1556
            [Valid] AUROC: 0.7847 / AUPRC: 0.1823
            Trial 9 - Epoch 20/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.62it/s]
            [Train] AUROC: 0.7525 / AUPRC: 0.1608
            [Valid] AUROC: 0.7907 / AUPRC: 0.1924
            Trial 9 - Epoch 21/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 64.34it/s]
            [Train] AUROC: 0.7775 / AUPRC: 0.1792
            [Valid] AUROC: 0.8059 / AUPRC: 0.1907
            Trial 9 - Epoch 22/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.20it/s]
            [Train] AUROC: 0.7950 / AUPRC: 0.1980
            [Valid] AUROC: 0.8193 / AUPRC: 0.1958
            Trial 9 - Epoch 23/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.56it/s]
            [Train] AUROC: 0.8078 / AUPRC: 0.2022
            [Valid] AUROC: 0.8223 / AUPRC: 0.1904
            Epoch 00023: reducing learning rate of group 0 to 4.6372e-05.
            Trial 9 - Epoch 24/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.03it/s]
            [Train] AUROC: 0.8153 / AUPRC: 0.2155
            [Valid] AUROC: 0.8257 / AUPRC: 0.1936
            Trial 9 - Epoch 25/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.84it/s]
            [Train] AUROC: 0.8177 / AUPRC: 0.2164
            [Valid] AUROC: 0.8158 / AUPRC: 0.1807
            Trial 9 - Epoch 26/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 65.49it/s]
            [Train] AUROC: 0.8228 / AUPRC: 0.2232
            [Valid] AUROC: 0.8285 / AUPRC: 0.1968
            Trial 9 - Epoch 27/47 - Training: 100%|██████████| 545/545 [00:08<00:00, 63.39it/s]
            [Train] AUROC: 0.8212 / AUPRC: 0.2156
            [I 2023-12-12 16:14:52,226] Trial 8 finished with value: 0.20010589072051646 and parameters: {'learning_rate': 0.0004637199702843728, 'batch_size': 32, 'num_epochs': 47, 'optimizer': 'SGD'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.8266 / AUPRC: 0.1923
            Early stopping
            > 
            
            ---
            
            ## <Best AUPRC Model's Performance>
            [train_loss]: 0.2438
            [valid_loss]: 0.3054
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.6443
            [valid_auroc]: 0.7690
            [train_auprc]: 0.1118
            [valid_auprc]: 0.2001 <- Pick It Up!
            
            ## [Learning Rate] : 0.000104941455994481
            [Batch Size] : 128
            [Epochs] : 48
            [Optimizer] : Adam (
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.000104941455994481
            maximize: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 10 - Epoch 1/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.10it/s]
            
            > [Train] AUROC: 0.5002 / AUPRC: 0.0692
            [Valid] AUROC: 0.4909 / AUPRC: 0.0739
            Trial 10 - Epoch 2/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.40it/s]
            [Train] AUROC: 0.4995 / AUPRC: 0.0685
            [Valid] AUROC: 0.4688 / AUPRC: 0.0680
            Trial 10 - Epoch 3/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.98it/s]
            [Train] AUROC: 0.4948 / AUPRC: 0.0672
            [Valid] AUROC: 0.4538 / AUPRC: 0.0634
            Trial 10 - Epoch 4/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.00it/s]
            [Train] AUROC: 0.5012 / AUPRC: 0.0704
            [Valid] AUROC: 0.4476 / AUPRC: 0.0647
            Trial 10 - Epoch 5/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.01it/s]
            [Train] AUROC: 0.5023 / AUPRC: 0.0697
            [Valid] AUROC: 0.3911 / AUPRC: 0.0540
            Trial 10 - Epoch 6/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.96it/s]
            [Train] AUROC: 0.5210 / AUPRC: 0.0733
            [Valid] AUROC: 0.4315 / AUPRC: 0.0576
            Trial 10 - Epoch 7/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.96it/s]
            [Train] AUROC: 0.5075 / AUPRC: 0.0691
            [Valid] AUROC: 0.2996 / AUPRC: 0.0476
            Epoch 00007: reducing learning rate of group 0 to 1.0494e-05.
            Trial 10 - Epoch 8/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.97it/s]
            [Train] AUROC: 0.5503 / AUPRC: 0.0801
            [Valid] AUROC: 0.4007 / AUPRC: 0.0558
            Trial 10 - Epoch 9/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.99it/s]
            [Train] AUROC: 0.5476 / AUPRC: 0.0783
            [Valid] AUROC: 0.4974 / AUPRC: 0.0681
            Trial 10 - Epoch 10/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.46it/s]
            [Train] AUROC: 0.5529 / AUPRC: 0.0815
            [Valid] AUROC: 0.5521 / AUPRC: 0.0732
            Trial 10 - Epoch 11/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.11it/s]
            [Train] AUROC: 0.5504 / AUPRC: 0.0814
            [Valid] AUROC: 0.5951 / AUPRC: 0.0849
            Trial 10 - Epoch 12/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.46it/s]
            [Train] AUROC: 0.5739 / AUPRC: 0.0855
            [Valid] AUROC: 0.5484 / AUPRC: 0.0735
            Trial 10 - Epoch 13/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.05it/s]
            [Train] AUROC: 0.5853 / AUPRC: 0.0909
            [Valid] AUROC: 0.7067 / AUPRC: 0.1205
            Trial 10 - Epoch 14/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.88it/s]
            [Train] AUROC: 0.6074 / AUPRC: 0.0951
            [Valid] AUROC: 0.6344 / AUPRC: 0.0936
            Trial 10 - Epoch 15/48 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.60it/s]
            [Train] AUROC: 0.6066 / AUPRC: 0.0983
            [Valid] AUROC: 0.7154 / AUPRC: 0.1253
            Trial 10 - Epoch 16/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.57it/s]
            [Train] AUROC: 0.6147 / AUPRC: 0.0986
            [Valid] AUROC: 0.7331 / AUPRC: 0.1317
            Trial 10 - Epoch 17/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.51it/s]
            [Train] AUROC: 0.6454 / AUPRC: 0.1147
            [Valid] AUROC: 0.7574 / AUPRC: 0.1562
            Trial 10 - Epoch 18/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.93it/s]
            [Train] AUROC: 0.6693 / AUPRC: 0.1235
            [Valid] AUROC: 0.7720 / AUPRC: 0.1678
            Trial 10 - Epoch 19/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.94it/s]
            [Train] AUROC: 0.6864 / AUPRC: 0.1292
            [Valid] AUROC: 0.7854 / AUPRC: 0.2021
            Trial 10 - Epoch 20/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.95it/s]
            [Train] AUROC: 0.7093 / AUPRC: 0.1562
            [Valid] AUROC: 0.7965 / AUPRC: 0.1939
            Trial 10 - Epoch 21/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.18it/s]
            [Train] AUROC: 0.7410 / AUPRC: 0.1720
            [Valid] AUROC: 0.7970 / AUPRC: 0.1897
            Trial 10 - Epoch 22/48 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.58it/s]
            [Train] AUROC: 0.7652 / AUPRC: 0.1963
            [Valid] AUROC: 0.7930 / AUPRC: 0.2532
            Trial 10 - Epoch 23/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.57it/s]
            [Train] AUROC: 0.7857 / AUPRC: 0.2176
            [Valid] AUROC: 0.8437 / AUPRC: 0.2901
            Trial 10 - Epoch 24/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.55it/s]
            [Train] AUROC: 0.8080 / AUPRC: 0.2514
            [Valid] AUROC: 0.8310 / AUPRC: 0.2645
            Trial 10 - Epoch 25/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.51it/s]
            [Train] AUROC: 0.8213 / AUPRC: 0.2800
            [Valid] AUROC: 0.7930 / AUPRC: 0.2668
            Trial 10 - Epoch 26/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.17it/s]
            [Train] AUROC: 0.8409 / AUPRC: 0.3118
            [Valid] AUROC: 0.8267 / AUPRC: 0.3059
            Trial 10 - Epoch 27/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.51it/s]
            [Train] AUROC: 0.8549 / AUPRC: 0.3464
            [Valid] AUROC: 0.8253 / AUPRC: 0.2482
            Trial 10 - Epoch 28/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.37it/s]
            [Train] AUROC: 0.8707 / AUPRC: 0.3768
            [Valid] AUROC: 0.8913 / AUPRC: 0.2992
            Trial 10 - Epoch 29/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.28it/s]
            [Train] AUROC: 0.8902 / AUPRC: 0.4040
            [Valid] AUROC: 0.9030 / AUPRC: 0.3040
            Trial 10 - Epoch 30/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.44it/s]
            [Train] AUROC: 0.8973 / AUPRC: 0.4380
            [Valid] AUROC: 0.7919 / AUPRC: 0.1407
            Trial 10 - Epoch 31/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.46it/s]
            [Train] AUROC: 0.9088 / AUPRC: 0.4402
            [Valid] AUROC: 0.7938 / AUPRC: 0.1416
            Trial 10 - Epoch 32/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.36it/s]
            [Train] AUROC: 0.9163 / AUPRC: 0.4616
            [Valid] AUROC: 0.7711 / AUPRC: 0.1293
            Epoch 00032: reducing learning rate of group 0 to 1.0494e-06.
            Trial 10 - Epoch 33/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.26it/s]
            [Train] AUROC: 0.9263 / AUPRC: 0.4822
            [Valid] AUROC: 0.8352 / AUPRC: 0.1711
            Trial 10 - Epoch 34/48 - Training: 100%|██████████| 137/137 [00:06<00:00, 19.62it/s]
            [Train] AUROC: 0.9243 / AUPRC: 0.4766
            [Valid] AUROC: 0.8626 / AUPRC: 0.1989
            Trial 10 - Epoch 35/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 19.20it/s]
            [Train] AUROC: 0.9250 / AUPRC: 0.4950
            [Valid] AUROC: 0.8270 / AUPRC: 0.1644
            Trial 10 - Epoch 36/48 - Training: 100%|██████████| 137/137 [00:07<00:00, 18.91it/s]
            [Train] AUROC: 0.9258 / AUPRC: 0.5055
            [I 2023-12-12 16:19:25,165] Trial 9 finished with value: 0.30587177223092793 and parameters: {'learning_rate': 0.000104941455994481, 'batch_size': 128, 'num_epochs': 48, 'optimizer': 'Adam'}. Best is trial 4 with value: 0.9213366982167329.
            [Valid] AUROC: 0.8627 / AUPRC: 0.1979
            Early stopping
            > 
            
            ---
            
            <Best AUPRC Model's Performance>
            [train_loss]: 0.2477
            [valid_loss]: 0.4583
            [train_accuracy]: 0.9306
            [valid_accuracy]: 0.9303
            [train_auroc]: 0.8409
            [valid_auroc]: 0.8267
            [train_auprc]: 0.3118
            [valid_auprc]: 0.3059 <- Pick It Up!
            최적의 하이퍼파라미터: {'learning_rate': 0.0004178442199935877, 'batch_size': 64, 'num_epochs': 36, 'optimizer': 'Adam'}
            
        
        - 1D CNN을 가지고, 출력코드만 수정 후 재 실험
            
            ## [I 2023-12-12 16:54:46,444] A new study created in memory with name: no-name-292dcb64-1403-45e4-8308-e1db35482461
            
            > Trial{trial.number+1} Start...<<<<<
            > 
            
            ---
            
            ## [Learning Rate] : 0.006970954061423948
            [Batch Size] : 64
            [Epochs] : 47
            [Optimizer] : Adam (
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.006970954061423948
            maximize: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 1 - Epoch 1/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.53it/s]
            
            > [Train] AUROC: 0.7425 / AUPRC: 0.1524
            [Valid] AUROC: 0.9312 / AUPRC: 0.3934
            Trial 1 - Epoch 2/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.90it/s]
            [Train] AUROC: 0.9327 / AUPRC: 0.4620
            [Valid] AUROC: 0.9607 / AUPRC: 0.5435
            Trial 1 - Epoch 3/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.61it/s]
            [Train] AUROC: 0.9545 / AUPRC: 0.5538
            [Valid] AUROC: 0.9715 / AUPRC: 0.6110
            Trial 1 - Epoch 4/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.97it/s]
            [Train] AUROC: 0.9606 / AUPRC: 0.5852
            [Valid] AUROC: 0.9768 / AUPRC: 0.6127
            Trial 1 - Epoch 5/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.10it/s]
            [Train] AUROC: 0.9691 / AUPRC: 0.6409
            [Valid] AUROC: 0.9773 / AUPRC: 0.6533
            Trial 1 - Epoch 6/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.66it/s]
            [Train] AUROC: 0.9688 / AUPRC: 0.6340
            [Valid] AUROC: 0.9769 / AUPRC: 0.6338
            Trial 1 - Epoch 7/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.09it/s]
            [Train] AUROC: 0.9701 / AUPRC: 0.6321
            [Valid] AUROC: 0.9769 / AUPRC: 0.6673
            Trial 1 - Epoch 8/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.97it/s]
            [Train] AUROC: 0.9724 / AUPRC: 0.6540
            [Valid] AUROC: 0.9784 / AUPRC: 0.6350
            Trial 1 - Epoch 9/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.70it/s]
            [Train] AUROC: 0.9765 / AUPRC: 0.6877
            [Valid] AUROC: 0.9809 / AUPRC: 0.7135
            Trial 1 - Epoch 10/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.29it/s]
            [Train] AUROC: 0.9768 / AUPRC: 0.7029
            [Valid] AUROC: 0.9765 / AUPRC: 0.6953
            Trial 1 - Epoch 11/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.30it/s]
            [Train] AUROC: 0.9779 / AUPRC: 0.7054
            [Valid] AUROC: 0.9821 / AUPRC: 0.6956
            Trial 1 - Epoch 12/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.46it/s]
            [Train] AUROC: 0.9784 / AUPRC: 0.7119
            [Valid] AUROC: 0.9786 / AUPRC: 0.6507
            Trial 1 - Epoch 13/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.83it/s]
            [Train] AUROC: 0.9800 / AUPRC: 0.7491
            [Valid] AUROC: 0.9821 / AUPRC: 0.6843
            Trial 1 - Epoch 14/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.00it/s]
            [Train] AUROC: 0.9805 / AUPRC: 0.7445
            [Valid] AUROC: 0.9692 / AUPRC: 0.6556
            Trial 1 - Epoch 15/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.02it/s]
            [Train] AUROC: 0.9811 / AUPRC: 0.7530
            [Valid] AUROC: 0.9801 / AUPRC: 0.6771
            Epoch 00015: reducing learning rate of group 0 to 6.9710e-04.
            Trial 1 - Epoch 16/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.90it/s]
            [Train] AUROC: 0.9838 / AUPRC: 0.7454
            [Valid] AUROC: 0.9815 / AUPRC: 0.7035
            Trial 1 - Epoch 17/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.03it/s]
            [Train] AUROC: 0.9850 / AUPRC: 0.7780
            [Valid] AUROC: 0.9817 / AUPRC: 0.7014
            Trial 1 - Epoch 18/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.89it/s]
            [Train] AUROC: 0.9845 / AUPRC: 0.7937
            [Valid] AUROC: 0.9801 / AUPRC: 0.7082
            Trial 1 - Epoch 19/47 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.72it/s]
            [Train] AUROC: 0.9863 / AUPRC: 0.7813
            [I 2023-12-12 16:57:21,450] Trial 0 finished with value: 0.7135198256772477 and parameters: {'learning_rate': 0.006970954061423948, 'num_epochs': 47}. Best is trial 0 with value: 0.7135198256772477.
            [Valid] AUROC: 0.9795 / AUPRC: 0.7030
            Early stopping
            > 
            
            ---
            
            ## < Trial {trial.number+1}'s Best Performance>
            [train_loss]: 0.0900
            [valid_loss]: 0.1026
            [train_accuracy]: 0.9601
            [valid_accuracy]: 0.9569
            [train_auroc]: 0.9765
            [valid_auroc]: 0.9809
            [train_auprc]: 0.6877
            [valid_auprc]: 0.7135 <- Pick It Up!
            
            > Trial{trial.number+1} Start...<<<<<
            > 
            
            ---
            
            ## [Learning Rate] : 0.00023198797065500428
            [Batch Size] : 64
            [Epochs] : 36
            [Optimizer] : Adam (
            Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.00023198797065500428
            maximize: False
            weight_decay: 0
            )
            
            > Let's GO!!! <<<<<
            > 
            
            ---
            
            Trial 2 - Epoch 1/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.96it/s]
            
            > [Train] AUROC: 0.5033 / AUPRC: 0.0714
            [Valid] AUROC: 0.4770 / AUPRC: 0.0703
            Trial 2 - Epoch 2/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.65it/s]
            [Train] AUROC: 0.5071 / AUPRC: 0.0711
            [Valid] AUROC: 0.4188 / AUPRC: 0.0607
            Trial 2 - Epoch 3/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.64it/s]
            [Train] AUROC: 0.4948 / AUPRC: 0.0681
            [Valid] AUROC: 0.4676 / AUPRC: 0.0662
            Trial 2 - Epoch 4/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.67it/s]
            [Train] AUROC: 0.6769 / AUPRC: 0.1313
            [Valid] AUROC: 0.8137 / AUPRC: 0.2226
            Trial 2 - Epoch 5/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.39it/s]
            [Train] AUROC: 0.8779 / AUPRC: 0.3637
            [Valid] AUROC: 0.9715 / AUPRC: 0.5868
            Trial 2 - Epoch 6/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.58it/s]
            [Train] AUROC: 0.9543 / AUPRC: 0.5927
            [Valid] AUROC: 0.9791 / AUPRC: 0.6442
            Trial 2 - Epoch 7/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.12it/s]
            [Train] AUROC: 0.9656 / AUPRC: 0.6662
            [Valid] AUROC: 0.9805 / AUPRC: 0.6447
            Trial 2 - Epoch 8/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.19it/s]
            [Train] AUROC: 0.9738 / AUPRC: 0.6751
            [Valid] AUROC: 0.9689 / AUPRC: 0.5739
            Trial 2 - Epoch 9/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.15it/s]
            [Train] AUROC: 0.9734 / AUPRC: 0.7254
            [Valid] AUROC: 0.9816 / AUPRC: 0.6953
            Trial 2 - Epoch 10/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.16it/s]
            [Train] AUROC: 0.9772 / AUPRC: 0.7342
            [Valid] AUROC: 0.9832 / AUPRC: 0.7056
            Trial 2 - Epoch 11/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.22it/s]
            [Train] AUROC: 0.9768 / AUPRC: 0.7419
            [Valid] AUROC: 0.9766 / AUPRC: 0.7368
            Trial 2 - Epoch 12/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.22it/s]
            [Train] AUROC: 0.9771 / AUPRC: 0.7485
            [Valid] AUROC: 0.9805 / AUPRC: 0.7419
            Trial 2 - Epoch 13/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.98it/s]
            [Train] AUROC: 0.9813 / AUPRC: 0.7738
            [Valid] AUROC: 0.9824 / AUPRC: 0.8140
            Trial 2 - Epoch 14/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.81it/s]
            [Train] AUROC: 0.9811 / AUPRC: 0.7920
            [Valid] AUROC: 0.9874 / AUPRC: 0.7799
            Trial 2 - Epoch 15/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.70it/s]
            [Train] AUROC: 0.9822 / AUPRC: 0.8065
            [Valid] AUROC: 0.9797 / AUPRC: 0.8103
            Trial 2 - Epoch 16/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.57it/s]
            [Train] AUROC: 0.9843 / AUPRC: 0.7965
            [Valid] AUROC: 0.9820 / AUPRC: 0.8073
            Trial 2 - Epoch 17/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.21it/s]
            [Train] AUROC: 0.9841 / AUPRC: 0.8229
            [Valid] AUROC: 0.9834 / AUPRC: 0.8534
            Trial 2 - Epoch 18/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.32it/s]
            [Train] AUROC: 0.9845 / AUPRC: 0.8286
            [Valid] AUROC: 0.9789 / AUPRC: 0.8537
            Trial 2 - Epoch 19/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.78it/s]
            [Train] AUROC: 0.9855 / AUPRC: 0.8282
            [Valid] AUROC: 0.9876 / AUPRC: 0.8684
            Trial 2 - Epoch 20/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.82it/s]
            [Train] AUROC: 0.9842 / AUPRC: 0.8367
            [Valid] AUROC: 0.9763 / AUPRC: 0.8372
            Trial 2 - Epoch 21/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.38it/s]
            [Train] AUROC: 0.9876 / AUPRC: 0.8572
            [Valid] AUROC: 0.9875 / AUPRC: 0.8567
            Trial 2 - Epoch 22/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.01it/s]
            [Train] AUROC: 0.9865 / AUPRC: 0.8410
            [Valid] AUROC: 0.9848 / AUPRC: 0.8488
            Trial 2 - Epoch 23/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.88it/s]
            [Train] AUROC: 0.9889 / AUPRC: 0.8630
            [Valid] AUROC: 0.9781 / AUPRC: 0.8747
            Trial 2 - Epoch 24/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.79it/s]
            [Train] AUROC: 0.9887 / AUPRC: 0.8762
            [Valid] AUROC: 0.9832 / AUPRC: 0.8207
            Trial 2 - Epoch 25/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.77it/s]
            [Train] AUROC: 0.9882 / AUPRC: 0.8866
            [Valid] AUROC: 0.9731 / AUPRC: 0.7172
            Trial 2 - Epoch 26/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.02it/s]
            [Train] AUROC: 0.9910 / AUPRC: 0.8938
            [Valid] AUROC: 0.9876 / AUPRC: 0.8672
            Trial 2 - Epoch 27/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.74it/s]
            [Train] AUROC: 0.9907 / AUPRC: 0.8905
            [Valid] AUROC: 0.9805 / AUPRC: 0.7857
            Trial 2 - Epoch 28/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.95it/s]
            [Train] AUROC: 0.9889 / AUPRC: 0.8907
            [Valid] AUROC: 0.9843 / AUPRC: 0.8642
            Trial 2 - Epoch 29/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.08it/s]
            [Train] AUROC: 0.9917 / AUPRC: 0.8939
            [Valid] AUROC: 0.9871 / AUPRC: 0.8761
            Trial 2 - Epoch 30/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.01it/s]
            [Train] AUROC: 0.9910 / AUPRC: 0.9051
            [Valid] AUROC: 0.9827 / AUPRC: 0.8730
            Trial 2 - Epoch 31/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.94it/s]
            [Train] AUROC: 0.9943 / AUPRC: 0.9260
            [Valid] AUROC: 0.9755 / AUPRC: 0.8760
            Trial 2 - Epoch 32/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.00it/s]
            [Train] AUROC: 0.9944 / AUPRC: 0.9348
            [Valid] AUROC: 0.9846 / AUPRC: 0.8708
            Trial 2 - Epoch 33/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.01it/s]
            [Train] AUROC: 0.9923 / AUPRC: 0.9292
            [Valid] AUROC: 0.9883 / AUPRC: 0.8765
            Trial 2 - Epoch 34/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 36.01it/s]
            [Train] AUROC: 0.9936 / AUPRC: 0.9283
            [Valid] AUROC: 0.9772 / AUPRC: 0.8667
            Trial 2 - Epoch 35/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 35.89it/s]
            [Train] AUROC: 0.9937 / AUPRC: 0.9351
            [Valid] AUROC: 0.9827 / AUPRC: 0.8943
            Trial 2 - Epoch 36/36 - Training: 100%|██████████| 273/273 [00:07<00:00, 34.79it/s]
            [Train] AUROC: 0.9939 / AUPRC: 0.9420
            [I 2023-12-12 17:02:14,602] Trial 1 finished with value: 0.8976881142994987 and parameters: {'learning_rate': 0.00023198797065500428, 'num_epochs': 36}. Best is trial 1 with value: 0.8976881142994987.
            [Valid] AUROC: 0.9837 / AUPRC: 0.8977
            > 
            
            ---
            
            < Trial {trial.number+1}'s Best Performance>
            [train_loss]: 0.0361
            [valid_loss]: 0.0728
            [train_accuracy]: 0.9878
            [valid_accuracy]: 0.9789
            [train_auroc]: 0.9939
            [valid_auroc]: 0.9837
            [train_auprc]: 0.9420
            [valid_auprc]: 0.8977 <- Pick It Up!
            최고 성능을 보인 시도: 1
            최적의 하이퍼파라미터: {'learning_rate': 0.00023198797065500428, 'num_epochs': 36}
            
        
        [PTB Test] AUROC: 0.9762 / AUPRC: 0.8437 / Accuracy: 96.74% / Loss: 0.0922
        
        [SPH Test] AUROC: 0.9931 / AUPRC: 0.8255 / Accuracy: 97.62% / Loss: 0.0361
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20110.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20111.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20112.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20113.png)
        
    - 하이퍼파라미터 찾기
        
        최적의 하이퍼파라미터 조합을 찾는 과정은 복잡할 수 있으며, 시행착오와 체계적인 접근이 필요합니다. 다음은 최적의 조합을 찾기 위한 구체적인 방법들입니다
        
        1. **그리드 탐색 (Grid Search)**
            - 여러 하이퍼파라미터에 대해 가능한 모든 조합을 시험합니다.
            - 각 조합에 대해 모델을 학습하고, 성능을 평가하여 최적의 조합을 찾습니다.
            - 시간이 많이 소요되지만, 작은 데이터셋이나 하이퍼파라미터 공간에서는 효과적일 수 있습니다.
        2. **랜덤 탐색 (Random Search)**
            - 하이퍼파라미터의 값 범위를 지정하고, 이 범위에서 무작위로 조합을 선택하여 모델을 학습합니다.
            - 그리드 탐색보다 빠르며, 때때로 더 좋은 결과를 얻을 수 있습니다.
            - 탐색 공간이 클 때 유용합니다.
        3. **베이지안 최적화 (Bayesian Optimization**
            - 이전의 평가 결과를 기반으로 하이퍼파라미터의 가장 유망한 영역을 예측합니다.
            - 평가 횟수를 줄이면서도 높은 성능의 조합을 찾을 수 있습니다.
            - 복잡한 하이퍼파라미터 공간에서 효과적입니다.
        4. **교차 검증 (Cross-Validation)**
            - 데이터셋을 여러 부분으로 나누어, 일부는 훈련에 일부는 검증에 사용합니다.
            - 모델의 일반화 능력을 더 잘 평가할 수 있습니다.
            - 각 하이퍼파라미터 조합에 대해 교차 검증을 수행하여 성능을 비교합니다.
        5. **학습률 스케줄링 (Learning Rate Scheduling)**
            - 학습률을 동적으로 조절하여 초기에는 빠르게, 후반에는 느리게 학습하도록 합니다.
            - 예를 들어, 학습률 감소 (learning rate decay) 또는 사이클릭 학습률 (cyclical learning rates)을 사용할 수 있습니다.
        6. **모델의 성능 지표 모니터링**
            - 손실, 정확도, AUROC 등의 지표를 지속적으로 모니터링하며, 이를 기반으로 하이퍼파라미터를 조정합니다.
        7. **일반화 능력 향상**
            - 드롭아웃, 배치 정규화 등의 기법을 사용하여 모델의 일반화 능력을 향상시킬 수 있습니다.
        
        하이퍼파라미터 조정은 모델, 데이터, 사용 사례에 따라 다를 수 있으므로, 다양한 전략을 시도하고 결과를 비교하는 것이 중요합니다. 또한, 탐색 과정에서 자동화 도구를 사용하면 시간과 노력을 절약할 수 있습니다.
        
    - 하이퍼파라미터 자동화
        
        자동화 도구는 하이퍼파라미터 튜닝 과정을 자동화하여 효율적으로 최적의 하이퍼파라미터를 찾을 수 있게 도와주는 도구입니다. 이러한 도구들은 일반적으로 다양한 하이퍼파라미터 설정을 자동으로 시도하고, 각 설정에 대한 모델의 성능을 평가하여 최적의 조합을 찾아냅니다. 몇 가지 대표적인 자동화 도구로는 `Optuna`, `Hyperopt`, `Ray Tune` 등이 있습니다.
        
        | 특징 / 도구 | Optuna | Hyperopt | Ray Tune |
        | --- | --- | --- | --- |
        | 모델 복잡성 | 중간에서 높음 | 중간에서 높음 | 중간에서 높음 |
        | 데이터의 특성 | 다양한 데이터 유형에 적합 | 다양한 데이터 유형에 적합 | 다양한 데이터 유형에 적합 |
        | 연산 자원 활용 | 단일 또는 멀티 코어 최적화 | 단일 코어 또는 간단한 병렬 처리 | 대규모 분산 환경에서 효과적 |
        | 최적화 방법 | 베이지안 최적화 | TPE (Tree-structured Parzen Estimator) | 다양한 최적화 알고리즘 지원 |
        | 사용 용이성 | 직관적이고 사용하기 쉬움 | 사용하기 쉬움 | 설정이 복잡할 수 있으나, 매우 유연 |
        | 병렬 처리 | 지원 | 제한적 지원 | 강력한 병렬 처리 및 분산 처리 |
        | 적합한 사용 사례 | 다양한 데이터셋 및 모델에 적합 | 작은 규모의 데이터셋 및 모델에 적합 | 대규모 데이터셋 및 복잡한 모델에 적합 |
        
        이 표를 참고하면, 각 도구가 가지는 장단점과 특정 사용 사례에 대한 적합성을 파악할 수 있습니다. 예를 들어, 단일 머신에서 간단한 실험을 수행하려면 `Optuna`나 `Hyperopt`가 적합할 수 있으며, 대규모 분산 환경에서 복잡한 모델을 실험하려면 `Ray Tune`이 더 적합할 수 있습니다.
        
        ### Optuna:
        
        `Optuna`는 파이썬 기반의 베이지안 최적화 라이브러리로, 광범위한 하이퍼파라미터 공간을 효율적으로 탐색할 수 있게 도와줍니다.
        
        ```python
        import optuna
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        def objective(trial):
            # 하이퍼파라미터 설정
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        
            model = SimpleCNN().to(device)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        
            # 훈련 루프
            for epoch in range(10):
                # 모델 학습...
        
            # 검증 성능 평가
            val_accuracy = evaluate_model(model)
            return val_accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        print(study.best_params)
        
        ```
        
        ### Hyperopt:
        
        `Hyperopt`는 `Optuna`와 유사하게 최적화 과정을 자동화합니다. `Hyperopt`는 `TPE (Tree-structured Parzen Estimator)` 알고리즘을 사용하여 하이퍼파라미터 공간을 탐색합니다.
        
        ```python
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        
        def objective(params):
            model = SimpleCNN(params).to(device)
            # 모델 학습 및 평가...
            val_accuracy = evaluate_model(model)
            return {'loss': -val_accuracy, 'status': STATUS_OK}
        
        space = {
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
            'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop', 'SGD'])
        }
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        
        ```
        
        ### Ray Tune:
        
        `Ray Tune`는 대규모 분산 환경에서의 하이퍼파라미터 튜닝을 지원합니다. 다양한 최적화 알고리즘과 쉬운 병렬 처리 기능을 제공합니다.
        
        ```python
        from ray import tune
        
        def train_model(config):
            model = SimpleCNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            # 모델 학습 및 평가...
        
        analysis = tune.run(
            train_model,
            config={
                'lr': tune.loguniform(1e-5, 1e-1),
                'optimizer': tune.choice(['Adam', 'RMSprop', 'SGD'])
            },
            num_samples=10
        )
        
        print("Best config: ", analysis.best_config)
        
        ```
        
        이러한 자동화 도구들은 대규모 하이퍼파라미터 공간을 효율적으로 탐색하고, 최적화 과정에서 시간과 노력을 절약하는 데 도움을 줍니다. 하지만 모델의 복잡성, 데이터의 특성, 연산 자원 등에 따라 적합한 도구를 선택해야 합니다.
        
    - adam
        
        `optimizer **=**` `tf.train.AdamOptimizer(learning_rate**=**0.001,beta1**=**0.9,beta2**=**0.999,epsilon**=**1e**-**08).minimize(loss)`
        
        기존 RMSprop과 momentum과 다르게 M(t)와 V(t)가 바로 W(t+1) 수식에 들어가는 것이 아니라 와 가 들어갑니다. 이 부분을 논문에서는 바이어스가 수정된 값으로 변경하는 과정이라고 합니다. 이전에 저희가 알아야할 것은 초기 M(0)와 V(0)값이 0으로 초기화 되는데 시작값이 0이기 때문에 이동평균을 구하면 0으로 편향된 값추정이 발생할 수 있습니다. 특히 초기 감쇠 속도가 작은 경우 (즉, β가 1에 가까울 때)에 발생합니다. 이를 방지하기 위해 값을 나누어 바이어스 보정을 해줍니다. 은 M(t)와 V(t)의 기대값을 구하는 과정에서 찾을 수 있다고 하는데 수학적 지식이 부족하여 어떻게 발견되는지 설명드리기가 힘들 것 같습니다. 추가적으로 α=0.001, β1로는 0.9, β2로는 0.999, ϵ 으로는 10^-8 값이 가장 좋은 Default값이라고 **[논문](https://arxiv.org/pdf/1412.6980.pdf)**에 명시되어 있습니다.
        
    - 배치 사이즈 변경
        
        현재 배치 사이즈가 64인 경우, 다음과 같이 여러 방법으로 배치 사이즈를 조정해 볼 수 있습니다
        
        1. **배치 사이즈 감소**: 배치 사이즈를 줄이면 모델이 더 자주 업데이트되므로 더 미세한 학습이 가능해집니다. 하지만 너무 작은 배치 사이즈는 학습 시간을 늘리고, 학습 과정을 불안정하게 만들 수 있습니다. 예를 들어, 32 또는 16으로 줄여볼 수 있습니다.
        2. **배치 사이즈 증가**: 반대로 배치 사이즈를 늘리면 메모리 사용량이 증가하지만, 일반적으로 학습 속도가 빨라집니다. 큰 배치 사이즈는 더 안정적인 그래디언트 추정을 제공하지만, 과적합을 유발할 수 있습니다. 128 또는 256으로 증가시켜 볼 수 있습니다.
        
        배치 사이즈를 변경할 때 고려해야 할 사항은 다음과 같습니다
        
        - **메모리 사용량**: 큰 배치 사이즈는 더 많은 메모리를 사용합니다. 사용 가능한 메모리 용량을 고려하여 배치 사이즈를 설정해야 합니다.
        - **학습 속도 및 안정성**: 작은 배치 사이즈는 노이즈가 더 많은 그래디언트 업데이트를 제공하여 일반적으로 학습 속도가 느려질 수 있습니다. 큰 배치 사이즈는 보다 안정적인 업데이트를 제공하지만, 지역 최소점(local minima)에 갇힐 위험이 있을 수 있습니다.
        - **학습 곡선**: 배치 사이즈를 변경한 후 학습 곡선을 관찰하여 모델이 어떻게 반응하는지 확인하는 것이 중요합니다. 손실 및 정확도 추이를 통해 최적의 배치 사이즈를 결정할 수 있습니다.
        
        배치 사이즈를 변경하는 것은 실험적인 과정이므로, 다양한 크기를 시도해보고 학습 과정과 결과를 비교해보는 것이 좋습니다.
        
    - scheduler 종류
        
        [https://sanghyu.tistory.com/113](https://sanghyu.tistory.com/113)
        
    - scheduler 설정
        
        ```jsx
        import torch
        import torch.optim as optim
        from annie_cnn_model import Custom1DCNN
        # ... 나머지 코드 ...
        
        # 손실 함수 및 옵티마이저 설정
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
        
        def train_model():
            # ... 나머지 코드 ...
        
            for epoch in range(num_epochs):
                # ... 학습 루프 ...
        
                # ... 검증 루프 ...
        
                # 스케줄러 업데이트
                scheduler.step(val_loss)
        
                # Early Stopping 및 기타 로직
                # ...
        
        if __name__ == "__main__":
            train_model()
        ```
        
        1. **optimizer**: 학습률을 조정할 옵티마이저를 지정합니다. 이 경우에는 **`optimizer`** 객체를 사용합니다.
        2. **'min'**: 이 모드는 스케줄러가 '손실(loss)'을 추적할 때 사용됩니다. 'min' 모드는 손실이 감소하는 것을 기대하며, 손실이 더 이상 줄어들지 않을 때 학습률을 조정합니다. (반대로 'max' 모드는 지표가 증가하는 것을 기대할 때 사용합니다.)
        3. **patience=5**: 스케줄러가 학습률을 조정하기 전에 기다려야 하는 에포크 수입니다. 여기서 **`patience=5`**는 손실이 5개의 연속된 에포크 동안 개선되지 않을 때까지 기다린다는 것을 의미합니다.
        4. **factor=0.1**: 학습률을 줄이는 비율을 나타냅니다. **`factor=0.1`**은 현재 학습률에 0.1을 곱하여 새로운 학습률을 설정한다는 것을 의미합니다. 예를 들어, 현재 학습률이 0.01이라면 새 학습률은 0.001이 됩니다.
        5. **verbose=True**: 스케줄러가 학습률을 조정할 때마다 콘솔에 메시지를 출력하도록 합니다. 이를 통해 학습 과정 중 학습률이 언제 조정되었는지 확인할 수 있습니다.
        
        ---
        
        스케줄러의 `patience` 매개변수와 얼리 스톱의 `patience` 매개변수는 둘 다 모델 학습 과정에서 중요한 역할을 하지만, 목적이 다릅니다.
        
        1. **스케줄러의 `patience`**
            - 이 매개변수는 학습률 스케줄러와 관련이 있습니다. 특정 기간 동안 검증 손실이 개선되지 않을 때 학습률을 조정하는 데 사용됩니다.
            - 예를 들어, `patience`가 10으로 설정되어 있으면, 스케줄러는 검증 손실이 10개의 에포크 동안 개선되지 않을 때 학습률을 낮추는 조치를 취합니다.
        2. **얼리 스톱의 `patience`**
            - 얼리 스톱은 모델이 더 이상 개선되지 않을 때, 즉 과적합을 방지하기 위해 훈련을 조기에 중단하는 기법입니다.
            - 여기서 `patience`는 검증 성능이 개선되지 않는 에포크의 허용 개수를 의미합니다. 만약 얼리 스톱의 `patience`가 10으로 설정되어 있다면, 검증 성능이 10개의 에포크 동안 개선되지 않으면 학습을 멈추고 최상의 모델을 저장합니다.
        
        두 `patience` 모두 성능이 일정 기간 동안 개선되지 않을 때 어떤 조치를 취할지 결정하는 역할을 하지만, 스케줄러는 학습률을 조정하고 계속 학습을 진행하는 반면, 얼리 스톱은 학습을 완전히 중단시키고 최적의 상태를 저장하는 데 사용됩니다.
        
    - earlystopping 설정 (2가지 방식)
        
        # class 사용
        
        ```jsx
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        from sklearn.metrics import roc_auc_score
        import numpy as np
        from sklearn.metrics import accuracy_score
        from annie_data_prep import create_dataloaders
        from annie_cnn_model import Custom1DCNN
        import json
        import yaml
        import random
        import wandb
        
        # 모델 시드 고정 
        # ... 기존 코드 ...
        
        # Early Stopping을 위한 클래스 정의
        class EarlyStopping:
            def __init__(self, patience=5, verbose=False, delta=0):
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_auroc_min = float('-inf')
                self.delta = delta
        
            def __call__(self, val_auroc, model):
                score = val_auroc
        
                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_auroc, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_auroc, model)
                    self.counter = 0
        
            def save_checkpoint(self, val_auroc, model):
                if self.verbose:
                    print(f'Validation AUROC increased ({self.val_auroc_min:.6f} --> {val_auroc:.6f}). Saving model ...')
                torch.save(model.state_dict(), 'model.pth')
                self.val_auroc_min = val_auroc
        
        # ... 기존 코드 ...
        
        def train_model():
            # ... 기존 코드 ...
        
            early_stopping = EarlyStopping(patience=7, verbose=True)
        
            # 학습 루프
            for epoch in range(num_epochs):
                # ... 기존 코드 ...
        
                # Early Stopping 호출
                early_stopping(val_auroc, model)
        
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
            # 전체 학습 과정의 결과를 JSON 파일로 저장
            # ... 기존 코드 ...
        
        if __name__ == "__main__":
            # ... 기존 코드 ...
        ```
        
        ---
        
        # 짧은 코드
        
        ```jsx
        def train_model():
            # ... 기존 코드 ...
        
            best_auroc = float('-inf')
            epochs_no_improve = 0
            early_stop = False
            patience = 7  # 성능 개선이 없는 에포크 수의 한계
        
            # 학습 루프
            for epoch in range(num_epochs):
                # ... 기존 코드 ...
        
                # Early Stopping 체크
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 'model.pth')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print("Early stopping")
                        break
        
            # 전체 학습 과정의 결과를 JSON 파일로 저장
            # ... 기존 코드 ...
        
        if __name__ == "__main__":
            # ... 기존 코드 ...
        ```
        
    
- **성능 평가 및 변화에 대한 고려**
    - FP, FN찾기
        
        1) GPT에게 문의
        
        ```python
        import torch
        
        def find_fp_fn_indices(predictions, labels):
            """
            참고: predictions와 labels는 PyTorch Tensor 또는 NumPy 배열이어야 합니다.
        
            Args:
            - predictions: 모델의 예측 결과
            - labels: 실제 레이블
        
            Returns:
            - fp_indices: False Positive의 인덱스 목록
            - fn_indices: False Negative의 인덱스 목록
            """
            predictions = torch.argmax(predictions, dim=1)  # 다중 클래스 분류의 경우 예측 클래스 선택
            labels = labels.view(-1)  # 레이블을 1차원 벡터로 평탄화
        
            # FP (False Positive) 찾기
            fp_indices = torch.where((predictions == 1) & (labels == 0))[0]
        
            # FN (False Negative) 찾기
            fn_indices = torch.where((predictions == 0) & (labels == 1))[0]
        
            return fp_indices, fn_indices
        
        # 예제를 위한 가상의 데이터 생성
        predictions = torch.tensor([0.8, 0.2, 0.4, 0.6])  # 모델의 예측값 (확률로 가정)
        labels = torch.tensor([1, 0, 1, 1])  # 실제 레이블
        
        # FP와 FN의 인덱스 찾기
        fp_indices, fn_indices = find_fp_fn_indices(predictions, labels)
        
        # 결과 출력
        print("False Positive 인덱스:", fp_indices)
        print("False Negative 인덱스:", fn_indices)
        ```
        
        2) WHERE 함수 이용
        
        [https://stackoverflow.com/questions/69278640/how-can-i-get-an-index-for-true-positive-false-positive-true-negative-and-fal](https://stackoverflow.com/questions/69278640/how-can-i-get-an-index-for-true-positive-false-positive-true-negative-and-fal)
        
        [cikit-learn을](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) 사용하면 tp, fp, tn 및 fn의 개수를 쉽게 계산할 수 있습니다 . 그런데 tp, fp, tn, fn 각각에 대한 데이터 인덱스를 알고 싶습니다.
        
        예를 들어)
        
        ```python
        pred: [1, 1, 1, 0, 0, 0]
        true: [1, 0, 1, 0, 1, 0]
        tp = [0, 2]
        fp = [1]
        tn = [3, 5]
        fn = [4]
        
        어떻게 얻을 수 있나요?
        ```
        
         모든 답변에 감사드립니다.
        
        **답변**
        
        및 하나의 목록을 사용 np.array하고 결합하세요 . 고유한 조합을 만들려면 이진수 시스템을 사용하세요.truepred
        
        ```python
        tn: 00 = 0
        fp: 01 = 1
        fn: 10 = 2
        tp: 11 = 3
        ```
        
        이제 numpy.where함수를 사용하여 인덱스 목록을 가져올 수 있습니다.
        
        ```python
        import numpy as np
        pred = [1, 1, 1, 0, 0, 0]
        true = [1, 0, 1, 0, 1, 0]
        unq = np.array([x + 2*y for x, y in zip(pred, true)])
        tp = np.array(np.where(unq == 3)).tolist()[0]
        fp = np.array(np.where(unq == 1)).tolist()[0]
        tn = np.array(np.where(unq == 0)).tolist()[0]
        fn = np.array(np.where(unq == 2)).tolist()[0]
        ```
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20114.png)
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20115.png)
        
        [https://yihchun.com/papers/hccis20.pdf](https://yihchun.com/papers/hccis20.pdf)
        
    - 에폭마다 성능이 달라지는 것에 대하여
        
        에폭의 수가 달라짐에 따라 AUROC (Area Under the Receiver Operating Characteristic) 값이 달라지는 현상은 모델 학습 과정의 정상적인 특성입니다. 이는 다음과 같은 몇 가지 이유로 발생할 수 있습니다
        
        1. **모델 학습과정**: 각 에폭에서 모델은 훈련 데이터를 사용하여 학습하고, 이를 통해 가중치가 업데이트됩니다. 초기 에폭에서는 모델이 아직 충분히 학습되지 않았을 수 있으므로, 성능이 낮게 나타날 수 있습니다. 학습이 진행됨에 따라 모델이 데이터를 더 잘 이해하고 예측 성능이 향상되므로 AUROC 값이 증가할 수 있습니다.
        2. **과적합(Overfitting)**: 너무 많은 에폭 동안 학습을 진행하면 모델이 훈련 데이터에 과적합될 수 있습니다. 과적합된 모델은 훈련 데이터에는 잘 맞지만, 검증 및 테스트 데이터에 대해서는 일반화 성능이 떨어질 수 있습니다. 이 경우, AUROC 값이 어느 지점 이후로 감소할 수 있습니다.
        3. **데이터의 다양성과 복잡성**: 사용하는 데이터셋의 특성에 따라 모델이 다르게 학습할 수 있습니다. 데이터셋이 복잡하거나 다양한 패턴을 가지고 있다면, 모델이 이를 학습하는 데 더 많은 에폭이 필요할 수 있습니다.
        4. **학습률과 하이퍼파라미터**: 학습률과 같은 하이퍼파라미터 설정은 모델의 학습 속도와 성능에 큰 영향을 미칩니다. 적절한 학습률 설정은 모델이 최적화 과정에서 안정적으로 성능을 향상시킬 수 있도록 도와줍니다.
        5. **랜덤성**: 딥러닝 모델 학습 과정에는 초기 가중치 설정, 미니 배치의 선택 등에 의한 랜덤성이 포함되어 있습니다. 이러한 랜덤성으로 인해 모델의 학습 과정과 성능에 변동이 생길 수 있습니다.
        
        따라서, AUROC 값의 변화는 모델이 학습 과정에서 어떻게 성능이 변화하는지를 보여주는 중요한 지표입니다. 에폭 수를 조정하고, 과적합을 방지하며, 적절한 하이퍼파라미터를 설정하는 것은 모델의 최적 성능을 달성하는 데 중요한 요소입니다.
        
    - 에폭이 얼마 안지났는데 val loss증가
        
        에폭이 얼마 지나지 않았음에도 검증 손실(**`val loss`**)이 급격히 증가하는 현상은 몇 가지 가능한 원인이 있습니다. 이를 진단하고 해결하기 위해 여러 방법을 시도해볼 수 있습니다
        
        1. **학습률 조정**: 높은 학습률은 때때로 모델이 최적점을 "넘어서" 가게 만들 수 있습니다. 학습률을 낮춰보세요. 예를 들어, 0.01에서 0.005나 0.001로 조정해보는 것이 좋습니다.
        2. **과적합(Overfitting) 여부 확인**: 모델이 훈련 데이터에 과도하게 적응하면서 검증 데이터에서 성능이 떨어질 수 있습니다. 이를 방지하기 위해 드롭아웃(Dropout) 레이어를 추가하거나, 데이터 증강(Data Augmentation)을 시도해볼 수 있습니다.
        3. **배치 크기 조정**: 너무 크거나 작은 배치 크기는 학습 과정에 부정적인 영향을 미칠 수 있습니다. 배치 크기를 조정해보세요. 예를 들어, 현재보다 크거나 작게 설정해 실험해볼 수 있습니다.
        4. **모델 복잡도 검토**: 너무 복잡한 모델은 과적합을 유발할 수 있습니다. 모델의 레이어 수나 파라미터 수를 줄여보는 것도 한 방법입니다.
        5. **조기 종료(Early Stopping) 사용**: 검증 손실이 일정 기간 동안 개선되지 않을 경우 학습을 중단하도록 설정합니다. 이는 과적합을 방지하는 데 도움이 됩니다.
        6. **데이터의 정규화 및 전처리 확인**: 데이터가 적절히 전처리되고 정규화되었는지 확인해보세요. 잘못된 전처리는 모델 성능에 부정적인 영향을 미칠 수 있습니다.
        7. **손실 함수 및 최적화 기법 검토**: 사용 중인 손실 함수와 최적화 기법이 문제에 적합한지 확인해보세요. 때때로 다른 손실 함수나 최적화 알고리즘이 더 좋은 결과를 가져올 수 있습니다.
        8. **학습 과정 모니터링**: 학습 및 검증 손실뿐만 아니라 다른 성능 지표(예: 정확도, F1 점수)를 모니터링하며 전반적인 모델의 성능을 파악하세요.
        
        이러한 조정 사항들을 순차적으로 적용해보며 모델의 성능이 어떻게 변화하는지 관찰하는 것이 중요합니다. 때로는 여러 가지 요소를 동시에 변경하지 않고 한 번에 하나씩 조정하며 그 영향을 관찰하는 것이 더 효과적일 수 있습니다.
        

- **활성화 함수 및 데이터 정규화**
    - activation/leaky_relu
        
        딥러닝 모델에서 사용할 수 있는 다양한 활성화 함수가 있습니다. 각 활성화 함수는 모델의 특성에 따라 선택할 수 있으며, 다음은 몇 가지 주요 활성화 함수입니다:
        
        1. **ReLU (Rectified Linear Unit)**:
            - ReLU는 가장 일반적으로 사용되는 활성화 함수 중 하나로, 양수 입력에 대해 선형 함수를 적용하고 음수 입력에 대해 0을 출력합니다.
            - 장점: 계산이 효율적이며, 다양한 문제에 잘 작동할 수 있습니다.
            - 단점: 음수 입력에 대해서는 0을 출력하기 때문에 "죽은 뉴런" 문제가 발생할 수 있습니다.
        2. **Sigmoid 함수**:
            - Sigmoid 함수는 입력을 [0, 1] 범위로 압축하는 S 곡선 모양의 활성화 함수입니다.
            - 주로 이진 분류 문제에서 출력층에서 사용됩니다.
            - 단점: Vanishing Gradient 문제가 있을 수 있으며, 중심값이 0이 아니기 때문에 가중치 업데이트가 느릴 수 있습니다.
        3. **Tanh 함수**:
            - Tanh 함수는 입력을 [-1, 1] 범위로 압축하는 함수로, Sigmoid와 유사하지만 중심값이 0입니다.
            - 장점: Sigmoid에 비해 중심값이 0이기 때문에 학습이 빠를 수 있습니다.
            - 단점: 여전히 Vanishing Gradient 문제가 발생할 수 있습니다.
        4. **LeakyReLU (Leaky Rectified Linear Unit)**:
            - 이미 이전에 언급한 것처럼, LeakyReLU는 음수 입력에 대해 작은 기울기를 가지며, 양수 입력에 대해서는 선형 함수를 사용합니다.
        5. **ELU (Exponential Linear Unit)**:
            - ELU는 음수 입력에 대해 지수 함수를 사용하여 부드러운 곡선을 형성하는 활성화 함수입니다.
            - 장점: ReLU와 유사한 이점을 가지면서, 양수와 음수 입력 모두에서 부드럽게 동작합니다.
        6. **Swish 함수**:
            - Swish는 Sigmoid 함수와 비슷하지만, 양수 입력에 대해 Sigmoid를 사용하여 부드럽게 변환합니다.
            - 최근에 제안된 함수 중 하나로 실험적으로 좋은 성능을 보이는 경우가 있습니다.
        7. **Softmax 함수**:
            - Softmax 함수는 다중 클래스 분류에서 출력층에서 사용되며, 입력값을 확률 분포로 변환합니다.
        
        ---
        
        LeakyReLU는 일반적인 ReLU(Rectified Linear Unit) 활성화 함수의 변형 중 하나로, 입력값이 음수일 때 작은 기울기(양수의 기울기를 가진 작은 상수)를 가지는 함수입니다. 이러한 특성은 몇 가지 장점을 제공할 수 있습니다.
        
        1. **Vanishing Gradient 문제 완화**: LeakyReLU는 음수 입력에 대해 작은 기울기를 가지므로, 그라디언트가 소멸하는 문제를 완화할 수 있습니다. 이것은 모델 학습을 더 안정적으로 만들 수 있습니다.
        2. **Dead Neurons 방지**: LeakyReLU를 사용하면 ReLU 활성화 함수에서 발생할 수 있는 "죽은 뉴런" 문제를 방지할 수 있습니다. 죽은 뉴런은 학습 도중 가중치 업데이트를 받지 않는 뉴런을 나타내며, LeakyReLU는 음수 입력에 대해 계속해서 업데이트를 받도록 도와줍니다.
        3. **Non-Linearity 유지**: LeakyReLU는 비선형성(Non-Linearity)을 유지하면서도 선형성을 적용하는 ReLU의 장점을 계승합니다. 따라서 모델이 복잡한 패턴을 학습할 수 있습니다.
        
        심전도 데이터에 음수 값이 있는 경우 ReLU(Rectified Linear Unit)를 적용하면 주의가 필요합니다. ReLU는 양수 입력에 대해서는 선형 함수를 사용하고 음수 입력에 대해서는 0을 출력합니다. 따라서 음수 입력값이 있는 경우에는 다음과 같은 문제가 발생할 수 있습니다:
        
        1. **Activation Dead Zone (활성화 죽음 영역)**: 음수 입력값에 대해 0을 출력하기 때문에 해당 뉴런은 학습 도중에 업데이트를 받지 않을 수 있습니다. 이로 인해 네트워크 내에서 "죽은 뉴런"이 발생할 수 있으며, 이는 모델의 성능을 저하시킬 수 있습니다.
        2. **Vanishing Gradient (사라지는 그래디언트)**: 음수 입력에 대해 그라디언트가 0이 되면 역전파(backpropagation) 과정에서 해당 뉴런의 가중치가 업데이트되지 않습니다. 이로 인해 그라디언트가 사라지는 문제가 발생할 수 있으며, 모델이 학습을 잘 수행하지 못할 수 있습니다.
        
        심전도 데이터에서 음수 값이 있는 경우, LeakyReLU나 ELU와 같은 다른 활성화 함수를 고려하는 것이 좋습니다. LeakyReLU는 음수 입력에 대해 작은 기울기를 가지므로 활성화 죽음 영역을 완화하고, ELU는 부드러운 곡선을 형성하여 다양한 입력값에 대응할 수 있습니다.
        
        딥러닝 모델을 설계할 때 활성화 함수를 선택할 때 데이터의 특성을 고려하고 실험을 통해 최적의 활성화 함수를 찾는 것이 중요합니다.
        
    - leakyReLU 적용 후 배열 확인(데이터 정규화)
        
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20116.png)
        
        ---
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20117.png)
        
        ---
        
        ### 정규화 후
        
        - ptb 데이터는 정상
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20118.png)
        
        - sph 데이터는 0.0으로 수렴
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20119.png)
        
        ---
        
        ### 문제 파악
        
        **`RuntimeWarning: overflow encountered in multiply`**와 **`RuntimeWarning: overflow encountered in reduce`** 경고는 정규화 과정에서 오버플로우가 발생했음을 나타냅니다.
        
        이는 **`sph_signals`** 데이터의 평균(mean)과 표준편차(standard deviation)를 계산하는 과정에서 수치적 불안정성이 생긴 것으로 추정됩니다. 이러한 오버플로우는 데이터의 범위가 너무 넓거나 특정 값들이 너무 크기 때문에 발생할 수 있습니다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20120.png)
        
        → sph 가 ptb 보다 데이터 범위가 이상하게 넓다. 그래서 오버플로우가 발생한 것 같다. 
        
        ---
        
        ### 문제 해결
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20121.png)
        
        평균과 표준편차 계산 시 오버플로우를 방지하기 위해 데이터 타입을 **`float64`**와 같이 더 큰 범위를 가진 타입으로 변경할 수 있습니다.
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20122.png)
        
        ---
        
        **또 다른 문제 해결법**
        
        **클리핑(Clipping)**: 데이터에 극단적인 값이 있는 경우, 이를 제한하는 클리핑 과정을 적용할 수 있습니다. 예를 들어, 데이터를 특정 범위로 제한하면 오버플로우를 방지할 수 있습니다.
        
        ```jsx
        sph_signals_clipped = np.clip(sph_signals, -1e3, 1e3)
        mean = np.mean(sph_signals_clipped)
        std = np.std(sph_signals_clipped)
        sph_signals_normalized = (sph_signals_clipped - mean) / (std + 1e-7)
        ```
        
    - 활성화함수 추천
        - **Swish**: 최근 연구에서 좋은 성능을 보인 활성화 함수입니다. Swish는 **`x * sigmoid(βx)`** 형태로, 입력값에 따라 출력값이 부드럽게 변화합니다.
        - **ELU (Exponential Linear Unit)**: 음수 입력에 대해 지수적으로 감소하는 출력을 제공합니다. 이는 ReLU와 비슷한 효과를 가지면서 음수 입력에 대한 정보를 일부 보존합니다.
        - **PReLU (Parametric ReLU)**: LeakyReLU와 유사하지만, 음수 부분의 기울기가 학습 가능한 매개변수가 됩니다. 이는 데이터에 따라 더 유연하게 조정될 수 있습니다.
        

- **기타 에러 및 예외 처리**
    - RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        
        이 오류는 PyTorch 텐서에 `grad` 속성이 설정되어 있는 상태에서 `numpy()` 메소드를 직접 호출할 때 발생합니다. PyTorch에서는 텐서가 그래디언트 계산을 위해 추적(tracking)되고 있을 때, 직접적으로 NumPy 배열로 변환하는 것을 허용하지 않습니다. 이를 해결하기 위해서는 먼저 `.detach()` 메소드를 사용해 텐서에서 그래디언트 추적을 제거한 다음 NumPy 배열로 변환해야 합니다.
        
        예를 들어, 다음과 같은 코드에서
        
        ```python
        val_preds.extend(outputs.view(-1).cpu().numpy())
        ```
        
        `outputs` 텐서는 모델의 순전파 과정에서 생성되므로 그래디언트 계산을 위해 추적되고 있습니다. 따라서 NumPy 배열로 변환하기 전에 `.detach()`를 호출해야 합니다.
        
        ```python
        val_preds.extend(outputs.view(-1).cpu().detach().numpy())
        ```
        
        이렇게 하면 `RuntimeError: Can't call numpy() on Tensor that requires grad.` 오류를 방지할 수 있습니다.
        
        이 변경을 코드의 모든 부분에 적용해야 합니다. 예를 들어, 훈련 데이터 루프와 테스트 데이터 루프에서도 같은 방식으로 수정해야 합니다.
        
    - Spectrogram
        1. **Octave (음높이):**
            - **특징:** Octave는 음성이나 음악 분야에서 주파수를 나타내는 데 사용되는 단위입니다. 심박수 데이터를 octave로 표현하면 주파수 영역에서의 변화를 시각적으로 확인할 수 있습니다.
            - **적용 분야:** 주로 음악이나 음성 데이터의 주파수 분석에 사용됩니다. 의료 분야에서는 특별한 경우가 아니라면 일반적으로 사용되지 않습니다.
        2. **Spectrogram (스펙트로그램):**
            - **특징:** Spectrogram은 주파수와 시간에 따른 신호 강도를 나타내는 2D 이미지입니다. 시간 경과에 따른 주파수의 변화를 쉽게 확인할 수 있습니다.
            - **적용 분야:** 음성 처리, 음악 분석, 신호 처리 등에서 많이 사용됩니다. 의료 분야에서는 특히 심박수 데이터의 주기성과 변화를 보다 자세히 이해하고자 할 때 사용됩니다.
        3. **Mel Spectrogram:**
            - **특징:** Mel Spectrogram은 주파수를 Mel 스케일로 변환하여 표현한 Spectrogram입니다. 사람의 청각 특성에 더 가까운 특성을 가지고 있어, 음성 및 음악 처리에서 자주 사용됩니다.
            - **적용 분야:** 음성 및 음악 처리, 음질 분석, 음성 인식 등에서 활용됩니다. 의료 분야에서는 Mel Spectrogram 역시 심박수 데이터를 더 자세히 분석하는 데 활용될 수 있습니다.
            
            ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20123.png)
            
        
        ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20124.png)
        
    - image_kernel 계산 실습
        
        [https://setosa.io/ev/image-kernels/](https://setosa.io/ev/image-kernels/)
        

# ****[6] 실험관리****

- wandb 자료
    - 공식 문서 [https://docs.wandb.ai/guides/track](https://docs.wandb.ai/guides/track)
    - 간단 사용법 [https://mingyu6952.tistory.com/entry/wandb-사용법Pytorch-CIFAR10-분류](https://mingyu6952.tistory.com/entry/wandb-%EC%82%AC%EC%9A%A9%EB%B2%95Pytorch-CIFAR10-%EB%B6%84%EB%A5%98)
    - 공식 노트북 튜토리얼 [https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=vdzlCfP9oOMq](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=vdzlCfP9oOMq)
    
- wandb test 실행 후 오류
    
    Exception in thread NetStatThr:
    Exception in thread IntMsgThr:
    Traceback (most recent call last):
    Traceback (most recent call last):
    File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
    self.run()
    File "/usr/lib/python3.8/threading.py", line 870, in run
    File "/usr/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
    self._target(*self._args, **self._kwargs)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/wandb_run.py", line 268, in check_network_status
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/wandb_run.py", line 300, in check_internal_messages
    self._loop_check_status(
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/wandb_run.py", line 224, in _loop_check_status
    self._loop_check_status(
    local_handle = request()
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/wandb_run.py", line 224, in _loop_check_status
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface.py", line 766, in deliver_internal_messages
    local_handle = request()
    return self._deliver_internal_messages(internal_message)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface.py", line 758, in deliver_network_status
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_shared.py", line 490, in _deliver_internal_messages
    return self._deliver_network_status(status)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_shared.py", line 484, in _deliver_network_status
    return self._deliver_record(record)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_shared.py", line 437, in _deliver_record
    return self._deliver_record(record)
    handle = mailbox._deliver_record(record, interface=self)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_shared.py", line 437, in _deliver_record
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/mailbox.py", line 455, in _deliver_record
    handle = mailbox._deliver_record(record, interface=self)
    interface._publish(record)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/mailbox.py", line 455, in _deliver_record
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_sock.py", line 51, in _publish
    interface._publish(record)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/interface/interface_sock.py", line 51, in _publish
    self._sock_client.send_record_publish(record)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 221, in send_record_publish
    self._sock_client.send_record_publish(record)
    self.send_server_request(server_req)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 221, in send_record_publish
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 155, in send_server_request
    self._send_message(msg)
    self.send_server_request(server_req)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 152, in _send_message
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 155, in send_server_request
    self._sendall_with_error_handle(header + data)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 130, in _sendall_with_error_handle
    self._send_message(msg)
    sent = self._sock.send(data)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 152, in _send_message
    BrokenPipeError: [Errno 32] Broken pipe
    self._sendall_with_error_handle(header + data)
    File "/usr/local/lib/python3.8/dist-packages/wandb/sdk/lib/sock_client.py", line 130, in _sendall_with_error_handle
    sent = self._sock.send(data)
    BrokenPipeError: [Errno 32] Broken pipe
    
    ---
    
    1. **성능 모니터링**: wandb는 학습뿐만 아니라 테스트 과정에서의 성능도 모니터링할 수 있게 해줍니다. 이를 통해 모델이 어떻게 작동하는지, 어떤 데이터에 대해 잘 또는 못하는지 파악할 수 있습니다.
    2. **결과 비교**: 학습과 테스트 단계 모두에서 wandb를 사용하면, 이 두 단계의 결과를 쉽게 비교하고 분석할 수 있습니다. 이것은 모델의 일반화 능력을 평가하는 데 도움이 됩니다.
    3. **버그 추적**: 만약 모델이 예상대로 작동하지 않는다면, wandb 로그를 통해 문제의 원인을 찾아낼 수 있습니다. 예를 들어, 과적합이나 데이터 처리 오류 등을 파악할 수 있습니다.
    4. **실험 관리**: 여러 실험을 진행할 때, wandb를 사용하면 각 실험의 세부 설정과 결과를 기록하고 관리하기가 용이합니다.
    
- wandb 설정 후 실험
    
    **기존 1D CNN 모델 코드로 실행.**
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20125.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20126.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20127.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20128.png)
    
    ---
    
    # wandb
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20129.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20130.png)
    
    ![Untitled](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/Untitled%20131.png)
    
- broken pipe 오류 해결
    
    Exception in thread ChkStopThr:
    Traceback (most recent call last):
    Exception in thread IntMsgThr:
    File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    Traceback (most recent call last):
    self.run()
    File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    File "/usr/lib/python3.8/threading.py", line 870, in run
    self.run()
    File "/usr/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
    self._target(*self._args, **self._kwargs)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/wandb_run.py", line 286, in check_stop_status
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/wandb_run.py", line 300, in check_internal_messages
    self._loop_check_status(
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/wandb_run.py", line 224, in _loop_check_status
    local_handle = request()
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface.py", line 748, in deliver_stop_status
    self._loop_check_status(
    return self._deliver_stop_status(status)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/wandb_run.py", line 224, in _loop_check_status
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_shared.py", line 468, in _deliver_stop_status
    local_handle = request()
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface.py", line 764, in deliver_internal_messages
    return self._deliver_record(record)
    return self._deliver_internal_messages(internal_message)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_shared.py", line 437, in _deliver_record
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_shared.py", line 490, in _deliver_internal_messages
    handle = mailbox._deliver_record(record, interface=self)
    return self._deliver_record(record)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/mailbox.py", line 455, in _deliver_record
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_shared.py", line 437, in _deliver_record
    interface._publish(record)
    handle = mailbox._deliver_record(record, interface=self)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_sock.py", line 51, in _publish
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/mailbox.py", line 455, in _deliver_record
    self._sock_client.send_record_publish(record)
    interface._publish(record)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 221, in send_record_publish
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/interface/interface_sock.py", line 51, in _publish
    self.send_server_request(server_req)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 155, in send_server_request
    self._sock_client.send_record_publish(record)
    self._send_message(msg)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 221, in send_record_publish
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 152, in _send_message
    self.send_server_request(server_req)
    self._sendall_with_error_handle(header + data)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 155, in send_server_request
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 130, in _sendall_with_error_handle
    self._send_message(msg)
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 152, in _send_message
    sent = self._sock.send(data)
    self._sendall_with_error_handle(header + data)
    BrokenPipeError: [Errno 32] Broken pipe
    File "/root/harbud/ecg/myenv/lib/python3.8/site-packages/wandb/sdk/lib/sock_client.py", line 130, in _sendall_with_error_handle
    sent = self._sock.send(data)
    BrokenPipeError: [Errno 32] Broken pipe
    
    ---
    
    제공해주신 오류 메시지는 Python에서 **`wandb`** 라이브러리를 사용하는 중에 발생한 **`BrokenPipeError`**와 관련이 있습니다. **`BrokenPipeError: [Errno 32] Broken pipe`** 오류는 일반적으로 프로세스가 소켓이나 파이프를 통해 데이터를 보내려고 할 때, 다른 쪽 끝에서 연결이 닫혀 있을 때 발생합니다.
    
    이 오류의 원인과 해결 방법은 다음과 같습니다:
    
    1. **네트워크 문제**: **`wandb`**는 모델의 학습 과정을 원격 서버에 로깅하는데, 네트워크 문제로 인해 서버와의 연결이 끊어진 경우 이러한 오류가 발생할 수 있습니다. 네트워크 연결을 확인하고, 문제가 지속되면 네트워크 설정을 점검해야 합니다.
    2. **`wandb` 서버 문제**: **`wandb`** 서버 측에서 문제가 발생하여 클라이언트와의 연결이 끊어진 경우도 있습니다. 이 경우 서버 상태를 확인하거나 잠시 후 다시 시도해야 합니다.
    3. **프로그램의 다른 부분에서의 오류**: 때로는 프로그램의 다른 부분에서 발생한 오류로 인해 **`wandb`** 클라이언트가 비정상적으로 중단될 수 있습니다. 전체 로그를 확인하여 다른 오류 메시지가 있는지 확인하고, 문제가 있다면 그 부분을 해결해야 합니다.
    4. **리소스 제한**: 시스템 리소스가 부족하거나 **`wandb`** 프로세스에 할당된 리소스가 부족할 때도 이 오류가 발생할 수 있습니다. 시스템의 메모리 사용량과 CPU 사용률을 확인하고, 필요하다면 리소스를 늘려야 합니다.
    5. **`wandb` 클라이언트 버전**: 사용 중인 **`wandb`**의 버전이 오래되었거나 버그가 있는 경우 최신 버전으로 업데이트하는 것이 좋습니다.
    
    이러한 가능한 원인들을 확인하고 문제를 해결한 후에도 오류가 계속된다면, 추가적으로 **`wandb`**의 로깅 설정을 점검하거나 **`wandb`** 지원 팀에 문의할 수 있습니다.
    
    ---
    
    1. **리소스 제한 문제 해결 방법**:
        - **메모리 사용량과 CPU 사용률 확인**: 시스템의 메모리 사용량과 CPU 사용률을 모니터링하는 도구를 사용하여 현재 리소스 사용 상태를 확인합니다. 예를 들어, Linux 시스템에서는 `htop` 또는 `top` 명령어를 사용할 수 있습니다.
        - **리소스 할당량 증가**: 만약 메모리나 CPU 사용률이 매우 높다면, 해당 프로세스에 더 많은 리소스를 할당하거나, 불필요한 프로세스를 종료하여 리소스를 확보해야 합니다.
        - **배치 크기 조정**: 머신러닝 모델의 배치 크기를 줄이는 것도 하나의 방법입니다. 배치 크기를 줄이면 한 번에 처리해야 하는 데이터 양이 줄어들어 메모리 사용량이 감소합니다.
        - **동시 실행 프로세스 감소**: 학습 중에 다른 메모리 집약적인 작업을 최소화하거나 중단합니다.
        - **하드웨어 업그레이드**: 만약 계속해서 리소스 부족 문제가 발생한다면, 더 많은 메모리나 더 빠른 CPU를 갖춘 하드웨어로 업그레이드하는 것을 고려해야 할 수도 있습니다.
    2. **`wandb` 클라이언트 버전 업데이트 방법**:
        - **현재 버전 확인**: 먼저, 현재 설치된 `wandb`의 버전을 확인합니다. 이는 Python 쉘에서 다음 명령어를 통해 할 수 있습니다:
            
            ```python
            import wandb
            print(wandb.__version__)
            
            ```
            
        - **최신 버전으로 업데이트**: 만약 설치된 버전이 최신 버전이 아니라면, `pip`를 사용하여 `wandb`를 업데이트합니다. 터미널에서 다음 명령어를 실행합니다:
            
            ```bash
            pip install --upgrade wandb
            
            ```
            
        - **업데이트 후 재시작**: `wandb`를 업데이트한 후에는 Python 환경을 재시작하고, 필요한 경우 스크립트를 다시 실행하여 변경사항이 적용되었는지 확인합니다.
    
    이러한 방법을 통해 리소스 제한 문제와 `wandb` 클라이언트의 버전 문제를 해결할 수 있습니다. 문제가 지속되면, 구체적인 오류 메시지나 상황을 바탕으로 추가적인 진단이 필요할 수 있습니다.
    

# ****[7] 실험 데이터베이스****

[ASHTON](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/ASHTON%20ee64ed952974429dbd778632976458b0.md)

[ANNIE](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/ANNIE%20b94766e634274d66a8724f26e056abe9.md)

[JWON](%E1%84%92%E1%85%A1%E1%84%87%E1%85%A5%E1%84%83%E1%85%B3%20%E1%84%8E%E1%85%AC%E1%84%8C%E1%85%A9%E1%86%BC%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%205a5ba08a7d5940518438b92ad1c6e919/JWON%20a7ff6c287d8d48268ab9116287792bb1.md)