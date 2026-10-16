# Project Architecture

## 1. 방향

* 목표: 모바일 기기에서 safetensors → MNN 변환과 추론을 실행하는 C++ 라이브러리
* 현재 모델 범위: SD1.5
* Converter와 Runtime은 독립 라이브러리로 구성
* Expo 연결·외부 저장소 연동·배포 패키징은 추후 별도 프로젝트에서 결정

| 작업 | 환경 |
| --- | --- |
| 변환·템플릿 생성·테스트 | Docker |
| Runtime 개발·검증 | Windows 네이티브, OpenCL |
| 모바일 실기기 테스트 | Android, OpenCL |

> Windows는 개발 속도를 위한 환경. PC 검증 결과를 모바일 검증으로 간주하지 않음

## 2. 프로젝트 구조

```text
cpp/
├── CMakeLists.txt             # Converter·Runtime·CLI 빌드 진입점
├── .clang-format
├── .clang-tidy
├── converter/
│   ├── CMakeLists.txt
│   ├── include/sd15/          # 공개 변환 API
│   ├── src/                   # 변환·양자화·작업 관리
│   └── tests/                 # C++ 테스트
├── cli/
│   ├── CMakeLists.txt
│   ├── convert_main.cpp
│   └── infer_main.cpp
├── runtime/
│   ├── CMakeLists.txt
│   ├── include/sd15/          # 공개 동기 추론 API
│   ├── src/                   # C++ 토큰화·난수·MNN 실행·CFG·DDIM
│   └── tests/                 # 계산·토큰화·반복 생성 검증
└── cmake/
    └── MNN.cmake              # 준비된 MNN 라이브러리 연결

src/sd15/                      # Python export·템플릿·비교·추론 입력 준비
tests/                         # Python 테스트·CLI 통합 테스트
scripts/                       # 빌드·lint 실행
docker/                        # 변환·템플릿·테스트 이미지
docs/
```

### 2-1. 작업 경로

| 경로 | 용도 |
| --- | --- |
| `MNNRuntime/` | MNN 소스·빌드 결과 |
| `build/` | 프로젝트 C++ 빌드 결과. Docker 빌드에서는 컨테이너 내부 경로 |
| `models/` | 체크포인트 모델 |
| `artifacts/` | ONNX·템플릿·변환 모델·검증 보고서 |
| `img/` | 추론 이미지 |

* `MNNRuntime/`은 외부 엔진 작업 경로. 프로젝트 Runtime 소스와 구분
* 추론 엔진은 `cpp/runtime/`에서 관리

### 2-2. cpp Runtime

| 클래스 | 역할 | 소유하는 주요 객체 |
| --- | --- | --- |
| `Sd15Runtime` | 추론 순서·CFG/DDIM·진행·취소 관리 | `Impl` |
| `Sd15Runtime::Impl` | 추론 간 유지할 설정·상태 보관 | `RuntimeOptions`, `ClipTokenizer`, busy 상태 |
| `ClipTokenizer` | prompt → token IDs | 어휘·BPE 병합 규칙 |
| `ClipEncoder` | token IDs → embedding | `MnnModelSession` |
| `UNetDenoiser` | latent·timestep·embedding → noise | `MnnModelSession` |
| `VaeDecoder` | VAE scale 적용·latent → 최종 RGB8 이미지 | `MnnModelSession` |
| `PreviewDecoder` | predicted x0 → TAESD 프리뷰 | `MnnModelSession` |
| `MnnModelSession` | 모델 로딩·실행·텐서 복사 및 검증 | `MNN::Interpreter`와 그에 속한 세션 |

* 모델별 클래스는 입출력 계약과 전후처리를 담당하고, 각각 소유한 `MnnModelSession`에 공통 실행을 맡긴다. 상속 관계는 없다.
* `Sd15Runtime`은 모델 객체를 멤버로 보관하지 않는다. `generate()`에서 CLIP → UNet → VAE 순서로 생성하고 각 단계가 끝나면 해제한다.
* 프리뷰 모델은 첫 프리뷰에 생성하고 denoising 종료까지 UNet과 함께 유지한다.
* CFG/DDIM·난수·픽셀 변환은 `diffusion.cpp`의 함수로 구현한다.

### 2-3. cpp Converter

상태·자원 관리:
| 클래스·구조체 | 역할 | 보관하는 데이터·상태 |
| --- | --- | --- |
| `ConversionPlan` / `FilePlan` / `ConversionTask` | 파일별 복사·zero·FP 변환·HQQ 작업 표현 | 위치·크기·저장 형식·양자화 설정 |
| `File` | 파일 읽기·쓰기·닫기 | 파일 핸들·경로 |
| `OutputDirectory` | 임시 출력 소유·완료 시 공개·실패 시 정리 | `.partial` 경로·생성 파일 목록 |
| `Progress` | 진행 알림·취소 확인·남은 시간 추정 | 처리량·측정 시간, 콜백·취소 토큰 참조 |
| `HqqWorkers` | HQQ 그룹을 나누어 병렬 계산 | 작업 스레드·동기화 상태, 배치 버퍼 참조 |

함수 역할:
| 함수 | 역할 |
| --- | --- |
| `convert` | 검사·계획 생성·청크 변환·출력 공개를 조율하고 결과 반환 |
| `cleanup_partial` | 소유 표식을 검증하고 중단된 임시 출력 정리 |
| `build_conversion_plan` | manifest·원본 텐서 정보를 검증해 `ConversionPlan` 추론 |
| `encode_float_chunk` / `decode_float_chunk` | FP16·FP32 저장값 변환과 유한성 검사 |
| `quantize_hqq` / `encode_hqq_group` | 그룹별 HQQ 계산과 W8·min/scale 저장값 추론 |

* 실행 흐름은 입력 검사 → 변환 계획 생성 → 청크별 읽기·계산·쓰기 → 출력 공개다. 파일 I/O와 실행 조율은 `converter.cpp`에서 담당한다.
* 계산 함수는 파일 I/O·콜백·공유 가변 상태에 의존하지 않는다. 입력을 변경하지 않고 호출자의 출력 버퍼에 기록하며, 내부 동적 할당 없이 버퍼를 재사용한다.
* `ConversionPlan`은 가중치 본문이나 파일 핸들을 보관하지 않는다. JSON과 원본 텐서 인덱스는 계획 생성 후 실행 전에 해제한다.
* 파일·임시 출력·HQQ 작업 스레드는 변환 한 번의 수명에 맞춰 관리한다. 실패·취소 시 작업 스레드가 버퍼 사용을 끝낸 뒤 자원을 정리한다.
* `convert()`는 동기 API다. 호출 계층이 작업 실행·취소 토큰 수명·진행 전달을 관리하며, 변환·정리는 프로세스 내 공통 mutex로 상호 배제한다.

### 2-4. 현재 의존성

```text
sd15-convert → sd15_converter → nlohmann_json, Threads
sd15-infer → sd15_runtime → MNN, ICU, nlohmann_json
sd15-convert, sd15-infer → CLI11
```

* Converter는 MNNConvert·MNN Runtime에 의존하지 않는다.
* CLI11은 CLI에만 연결하며 Converter·Runtime 라이브러리에는 연결하지 않는다.
* MNN은 고정 버전으로 별도 빌드하고, JSON·CLI11·ICU·Catch2는 `cpp/vcpkg.json`의 baseline으로 관리한다.
