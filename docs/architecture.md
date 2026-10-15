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
│   ├── src/                  # C++ 토큰화·난수·MNN 실행·CFG·DDIM
│   └── tests/                # 계산·토큰화·반복 생성 검증
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
* Runtime이 prompt·seed에서 token IDs·초기 latent를 생성.

Runtime 내부 역할:

```text
Sd15Runtime                  # 생성 순서·CFG/DDIM·단계별 수명
├── ClipEncoder              # token IDs → embedding
├── UNetDenoiser              # latent·timestep·context → noise
└── VaeDecoder               # latent scale·디코딩·RGB8 변환
    각 모델 → MnnModelSession # MNN 설정·세션 소유·텐서 복사/검증
```

* `sd15_models.hpp/.cpp`: 모델별 입출력 계약. 공통 `ModelOptions`를 각각 전달한다.
* `mnn_model_session.hpp/.cpp`: 모델 종류를 모르는 공통 MNN 실행 처리.
* 공개 `RuntimeOptions`의 `clip`·`unet`·`vae`는 공통 `ModelOptions`다. 별도 내부 설정 구조체나 변환 함수 없이 모델에 직접 전달한다.
* 세 모델은 `generate()`의 단계별 지역 객체이며 동시에 상주하지 않는다. 설정·tokenizer만 Runtime 수명 동안 유지한다.

현재 의존성:

```text
sd15-convert → sd15_converter → nlohmann_json, Threads
sd15-infer → sd15_runtime → MNN, ICU, nlohmann_json
sd15-convert, sd15-infer → CLI11
```

* Converter는 MNNConvert·MNN Runtime 없이 실행
* JSON·CLI11·ICU·Catch2는 `cpp/vcpkg.json`의 baseline으로 관리. CMake는 기존 `VCPKG_ROOT`를 사용하며 패키지는 빌드 폴더에 설치
* CLI11은 `SD15_BUILD_CLI=ON`일 때만 선택. Converter·Runtime 라이브러리에는 연결하지 않음
* MNN은 기존 Git 고정 네이티브 빌드를 유지하며 vcpkg로 관리하지 않음. 토크나이저의 Unicode 처리는 Runtime 전용 ICU 의존성으로 분리
* CMake 선택: `SD15_BUILD_CONVERTER`(기본 ON), `SD15_BUILD_RUNTIME`(기본 OFF), `SD15_BUILD_CLI`(기본 ON)
* 테스트는 `BUILD_TESTING`으로 선택. 기존 Docker Converter 빌드는 유지
* C++ 테스트는 vcpkg의 Catch2를 사용. 순수 단위 테스트는 CTest에 케이스별로 등록하며 자원·임시 폴더가 필요한 통합 테스트는 기존 실행 경로를 유지
* Python 인터프리터는 Converter 테스트를 켰을 때 필요
* Runtime은 Windows OpenCL 동기 이미지 생성과 요청별 협력적 취소를 구현. 비동기·캐시는 미구현
* Runtime 빌드·실행·검증 범위: [Runtime 문서](runtime-api.md)
