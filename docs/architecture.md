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

### 1-1. Converter와 Runtime의 책임

> Converter가 보장한 모델의 정적 구조는 추론 도중 다시 검증하지 않는다. 하지만 Runtime 초기화에서 실제로 성립해야 하는 조건까지 무조건 제거하지는 않는다.

검사 수를 줄이는 것보다 각 조건을 보장하는 계층과 시점을 명확히 하는 것이 목적이다.

| 계층 | 담당하는 일 | 보장 범위 |
| --- | --- | --- |
| PC 템플릿 준비 | 모델별 고정 그래프·가중치 매핑·저장 레이아웃 준비, 기준 모델과 비교 | 배포할 템플릿과 모델별 실행 계약 |
| Converter | 원본 tensor 이름·shape·dtype·데이터 범위, manifest 구간·양자화 설정 검사; 청크 변환·출력 공개 | 제공된 템플릿에 따라 가중치를 복원한 MNN 파일 |
| Runtime | 모델·tokenizer 로딩, tensor 연결, 토큰화·추론·전후처리, 진행·프리뷰·취소 | 현재 환경에서의 실행과 실행 결과·오류 처리 |
| 호출 계층 | 모델에 맞는 변환·실행 경로 선택, 파일 준비, 작업 스레드·객체 수명·UI 전달 관리 | 올바른 자원과 요청을 각 라이브러리에 전달 |

* Converter는 MNN 세션을 실행하지 않는다. 변환 성공은 템플릿에 따른 파일 복원을 뜻하며, GPU 실행 성공이나 추론 중 수치 안정성까지 보장하지 않는다.
* `manifest.json`은 Converter의 복원 지시서다. Runtime은 이를 다시 해석하지 않고 모델별 코드에 정의된 실행 계약을 사용한다. Runtime용 JSON은 현재 생성하지 않는다.
* tokenizer와 TAESD는 Converter 출력에 포함되지 않는 별도 자원이다. 호출 계층이 준비하고 Runtime이 로딩·초기 연결 시 확인한다.
* 현재는 SD1.5를 지원한다. 향후 SDXL 등도 모델별 템플릿과 전용 Runtime을 추가하며 같은 책임 경계를 유지한다. 범용 모델 자동 판별·실행 구조는 미리 구현하지 않는다.

### 1-2. 검증 시점

| 시점 | 확인할 조건 |
| --- | --- |
| 변환 | 원본 가중치·변환 결과의 유한성, 템플릿과 tensor의 정합성, 읽기·쓰기·출력 완성 여부 |
| Runtime 초기화 | 모델 경로, MNN 버전·모델 실행 옵션, tokenizer 자원 |
| 모델 세션 초기화 | 모델 로딩·OpenCL 세션 생성, 입력 개수, 사용할 tensor의 이름·shape·dtype |
| 생성 요청 시작 | prompt·seed·steps·guidance·프리뷰 옵션, busy 상태 |
| 추론 도중 | 입력 복사 크기, 할당·복사·실행 실패, NaN/Inf, 취소·콜백 오류 |

* 모델 세션은 해당 단계에서 로딩할 때 초기화한다. 검사를 위해 CLIP·UNet·VAE를 동시에 메모리에 올리지 않는다. TAESD는 첫 프리뷰 시 초기화한다.
* 연결된 tensor는 세션 resize 없이 재사용하며 추론마다 이름·shape·dtype를 다시 검사하지 않는다.
* NaN/Inf는 가중치가 유한해도 계산 중 발생할 수 있으므로 Runtime 책임이다. 정적 구조의 중복 검사와 구분한다.
* 내부 버퍼 크기 검사는 생성·전달 경로에서 보장되는 경우에만 정리한다. 현재 유지한 경계 검사와 Debug assertion은 2-2절에 기록한다.

### 1-3. 테스트 책임

* 템플릿·Converter 테스트: 가중치 매핑·변환값·그래프 보존, 잘못된 입력, 출력 공개·실패 복구를 검증한다.
* Runtime 테스트: 토큰화·CFG/DDIM·난수·픽셀 변환, 입력 검증·NaN/Inf·진행·취소·재사용을 검증한다.
* 연결 테스트: 변환한 모델이 Runtime의 입출력 계약으로 실행되는지 Windows OpenCL에서 확인한다. Android는 별도 검증한다.
* 실행 중 검사를 초기화나 Debug assertion으로 옮겨도 해당 계약을 확인하는 개발 테스트는 유지한다. 외부 MNN의 내부 구현 자체는 테스트하지 않는다.

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
| `Sd15Runtime` | 추론 순서·CFG·진행·취소 관리 | `Impl` |
| `DdimScheduler` | leading 시간표·alpha·DDIM 갱신·프리뷰 예측 | 요청별 시간표·누적 alpha |
| `DpmScheduler` | Karras 시간표·DPM++ 2M 갱신·프리뷰 예측 | sigma·이전 x0·스텝 인덱스 |
| `Sd15Runtime::Impl` | 추론 간 유지할 설정·상태 보관 | `RuntimeOptions`, `ClipTokenizer`, busy 상태 |
| `ClipTokenizer` | prompt → token IDs | 어휘·BPE 병합 규칙 |
| `ClipEncoder` | token IDs → embedding | `MnnModelSession` |
| `UNetDenoiser` | latent·timestep·embedding → noise | `MnnModelSession` |
| `VaeDecoder` | VAE scale 적용·latent → 최종 RGB8 이미지 | `MnnModelSession` |
| `PreviewDecoder` | predicted x0 → TAESD 프리뷰 | `MnnModelSession` |
| `MnnModelSession` | 모델 로딩·실행·텐서 복사 및 검증 | `MNN::Interpreter`와 그에 속한 세션 |

* 모델별 클래스는 입출력 계약과 전후처리를 담당하고, 각각 소유한 `MnnModelSession`에 공통 실행을 맡긴다. 상속 관계는 없다.
* 모델 생성 시 tensor 이름·shape·dtype를 한 번 확인하고 연결을 보관한다. 추론 중에는 연결을 재사용하며 입력 버퍼 크기·복사·실행 오류를 검사한다. 세션 resize는 사용하지 않는다.
* `Sd15Runtime`은 모델 객체를 멤버로 보관하지 않는다. `generate()`에서 CLIP → UNet → VAE 순서로 생성하고 각 단계가 끝나면 해제한다.
* 프리뷰 모델은 첫 프리뷰에 생성하고 denoising 종료까지 UNet과 함께 유지한다.
* CFG·난수·픽셀 변환은 `diffusion.cpp`, 시간표·학습 노이즈·알고리즘별 계산은 각 scheduler에서 구현한다.
* DDIM의 기존 누적 alpha 연산은 보존한다. DPM은 생성자에서 학습 beta·alpha·sigma·log-sigma와 Karras 시간표를 계산한다. beta·alpha는 FP32, 누적곱·보간은 double이며 추론용 sigma는 float로 저장한다. 요청 중에는 완성한 시간표를 재사용하며 정적 데이터 표나 Python에 의존하지 않는다.
* DPM 테스트는 시간표의 범위·단조성·유한성과 오류 처리·요청별 상태 분리를 검증한다. Python 참조 JSON이나 생성 도구에 의존하지 않는다.
* Runtime은 scheduler의 시간표를 순회하며 입력 스케일링 → UNet·CFG → scheduler 갱신을 실행한다. 진행·취소·프리뷰 전달은 Runtime이 담당한다.
* `make_scheduler`는 `SamplingPreset`에 따라 요청별 `Scheduler` 전략을 생성한다. 현재 DDIM leading과 DPM++ 2M Karras만 지원하며 두 전략 모두 초기 노이즈 배율 1·항등 입력 스케일링을 사용한다.
* DPM의 이력은 요청에 귀속된다. 취소·실패·완료 후 다음 요청은 새 전략으로 시작한다. 시간표의 정수 timestep이 중복되어도 배열 인덱스로 이력을 진행한다.
* SD1.5 scheduler 설정은 고정 상수다. 요청 검증은 steps 범위만 확인하며 시간표를 만들지 않는다. NaN/Inf·실행 실패 검사는 Runtime에서 유지한다.

내부 버퍼 크기의 보장 경로:

| 버퍼 | 생성·크기 보장 | 소비·유지 |
| --- | --- | --- |
| tokens | `ClipTokenizer::encode`가 77개로 resize | CLIP 입력 복사 |
| context | CLIP 출력 `[1,77,768]`을 초기 연결에서 확인 | UNet까지 const 참조, 크기 변경 없음 |
| latent | `initial_noise`가 16,384개 생성 | DDIM은 원소만 갱신, VAE는 복사 후 원소만 scale |
| timestep | 호출마다 원소 하나로 생성 | UNet 입력 복사 |
| noise | UNet 출력 `[1,4,64,64]`을 초기 연결에서 확인 | CFG가 원소만 갱신하고 scheduler에 const 참조로 전달 |
| clean prediction | 프리뷰 스텝에서 latent 크기로 resize | 같은 스텝에서만 TAESD에 전달 |
| decoded | VAE·TAESD 출력 `[1,3,512,512]`을 초기 연결에서 확인 | TAESD 후처리는 원소만 갱신, `pixels`에서 RGB8 변환 |

* 출력 vector는 연결한 tensor의 host 복사에서 생성한다. 이 경로는 고정 shape이며 세션 resize를 하지 않는다는 전제다.
* CFG·DDIM의 동일 버퍼 크기, scheduler의 스텝 인덱스와 `pixels`의 출력 크기는 내부 전제조건이다. Debug assertion으로 확인하고 Release에서는 재검사하지 않는다.
* `MnnModelSession::input`은 가변 길이 vector를 고정 host 버퍼에 복사하는 경계다. 잘못된 내부 호출의 초과 쓰기·부족한 입력을 막는 크기 검사는 유지한다.

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
