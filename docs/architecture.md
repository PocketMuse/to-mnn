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
├── CMakeLists.txt             # Converter·Runtime·CLI 빌드 진입점
├── .clang-format
├── .clang-tidy
├── converter/
│   ├── CMakeLists.txt
│   ├── include/sd15/          # 공개 변환 API
│   ├── src/                  # 변환·양자화·작업 관리
│   └── tests/                # C++ 테스트
├── cli/
│   ├── CMakeLists.txt
│   ├── convert_main.cpp
│   └── infer_main.cpp
├── runtime/
│   ├── CMakeLists.txt
│   ├── include/sd15/          # 공개 동기 추론 API
│   ├── src/                  # MNN 실행·CFG·DDIM·RGB 변환
│   └── tests/                # 계산·입력 검증
└── cmake/
    └── MNN.cmake             # 준비된 MNN 라이브러리 연결

src/sd15/                     # Python export·템플릿·비교·추론 입력 준비
tests/                        # Python 테스트·CLI 통합 테스트
scripts/                      # 빌드·lint 실행
docker/                       # 변환·템플릿·테스트 이미지
docs/                         # 정책·API·검증 기록
```

로컬 작업 경로(Git 추적 제외):

| 경로 | 용도 |
| --- | --- |
| `MNNRuntime/` | 외부 MNN 소스·빌드 결과 |
| `build/` | 프로젝트 C++ 빌드 결과. Docker 빌드에서는 컨테이너 내부 경로 |
| `models/` | 입력 체크포인트 |
| `artifacts/` | ONNX·템플릿·변환 모델·검증 보고서 |
| `img/` | 추론 이미지 |

* `MNNRuntime/`은 외부 엔진 작업 경로. 프로젝트 Runtime 소스와 구분
* 추론 엔진은 `cpp/runtime/`에서 관리
* 입력 token IDs·초기 latent는 `src/sd15/prepare_inference.py`에서 준비

현재 의존성:

```text
sd15-convert → sd15_converter → nlohmann_json, Threads
sd15-infer → sd15_runtime → MNN
```

* Converter는 MNNConvert·MNN Runtime 없이 실행
* CMake 선택: `SD15_BUILD_CONVERTER`(기본 ON), `SD15_BUILD_RUNTIME`(기본 OFF), `SD15_BUILD_CLI`(기본 ON)
* 테스트는 `BUILD_TESTING`으로 선택. 기존 Docker Converter 빌드는 유지
* Python 인터프리터는 Converter 테스트를 켰을 때 필요
* Runtime은 Windows OpenCL 동기 이미지 생성까지 구현. 취소·비동기·캐시는 미구현
* Runtime 빌드·실행·검증 범위: [Runtime 문서](runtime-api.md)
