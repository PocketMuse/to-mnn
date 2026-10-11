# Project Architecture

## 1. 방향

* 목표: 모바일 기기에서 safetensors → MNN 변환과 추론을 실행하는 C++ 라이브러리
* 현재 모델 범위: SD1.5. 템플릿 변환은 이 범위에서 마무리하고 Runtime 개발로 진행
* Converter와 Runtime은 독립 라이브러리로 구성. 변환된 MNN 파일로 연결
* Expo 연결·외부 저장소 연동·배포 패키징은 추후 별도 프로젝트에서 결정

| 작업 | 환경 |
| --- | --- |
| 변환·템플릿 생성·테스트 | Docker |
| Runtime 개발·검증 | Windows 네이티브, OpenCL |
| 모바일 실기기 테스트 | Android, OpenCL |

> Windows는 개발 속도를 위한 환경. PC 검증 결과를 모바일 검증으로 간주하지 않음

## 2. 현재 구조

```text
cpp/
├── CMakeLists.txt             # Converter·CLI 빌드 진입점
├── .clang-format
├── .clang-tidy
├── converter/
│   ├── CMakeLists.txt
│   ├── include/sd15/          # 공개 변환 API
│   ├── src/                  # 변환·양자화·작업 관리
│   └── tests/                # C++ 테스트
├── cli/
│   ├── CMakeLists.txt
│   └── convert_main.cpp
└── pymnn/
    └── CMakeLists.txt         # 기존 Windows MNN의 Python 바인딩

src/sd15/                     # Python export·템플릿·비교·참조 추론
tests/                        # Python 테스트·CLI 통합 테스트
scripts/                      # 빌드·lint 실행
docker/                       # 변환·템플릿·테스트 이미지
docs/                         # 정책·API·검증 기록
```

로컬 작업 경로(Git 추적 제외):

| 경로 | 용도 |
| --- | --- |
| `MNNRuntime/` | 외부 MNN 소스·빌드 결과·Python 바인딩 |
| `build/` | 프로젝트 C++ 빌드 결과. Docker 빌드에서는 컨테이너 내부 경로 |
| `models/` | 입력 체크포인트 |
| `artifacts/` | ONNX·템플릿·변환 모델·검증 보고서 |
| `img/` | 추론 이미지 |

* `MNNRuntime/`은 외부 엔진 작업 경로. 향후 구현할 우리 Runtime 소스와 구분
* 현재 추론 흐름은 `src/sd15/infer_mnn.py`에 있음
* `cpp/runtime/`은 아직 없음

현재 의존성:

```text
sd15-convert → sd15_converter → nlohmann_json, Threads
Python 참조 추론 → pymnn → MNN
```

* Converter는 MNNConvert·MNN Runtime 없이 실행
* CMake는 현재 Converter와 CLI를 함께 구성. 테스트는 `BUILD_TESTING`으로 선택
* Python 인터프리터는 Converter 테스트를 켰을 때 필요. `pymnn`은 별도 빌드
