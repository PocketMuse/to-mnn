# SD1.5 Runtime API

## 1. 개요
prompt·negative prompt·seed로 이미지를 추론하는 동기 API

* 입력: SD1.5 MNN 모델·CLIP tokenizer, 추론 옵션
* 출력: seed와 512×512 RGB8 이미지
* 실행 환경: Windows 네이티브 MNN 3.6.1, OpenCL buffer
* 지원: batch 1, CFG + DDIM eta=0 또는 DPM++ 2M Karras. 해상도는 고정
* 공개 헤더: [runtime.hpp](../cpp/runtime/include/sd15/runtime.hpp)
* 관련 문서: [아키텍처](architecture.md#2-2-cpp-runtime), [양자화 정책·검증](sd15.md), [모델 준비](../README.md)
* Windows 빌드: [MNN 준비](../README.md#3-mnnruntime-빌드) 후 Git Bash에서 `bash scripts/build-runtime.sh` 실행. `VCPKG_ROOT` 필요

## 2. API 목록

`sd15` 네임스페이스 기준이다.

| API | 용도 |
| --- | --- |
| `Sd15Runtime` 생성자·소멸자 | 실행 설정·tokenizer 준비와 자원 해제 |
| `Sd15Runtime::generate` | 동기 이미지 추론·진행·프리뷰 전달 |
| `Sd15Runtime::generate` 취소 오버로드 | 취소 토큰을 사용하는 동기 이미지 추론 |
| `GenerationCancellationToken::request_cancel` / `is_requested` | 취소 요청·확인 |
| `GenerationCancelled::what` | 취소 예외 설명 |

## 3. 공통 입력

### 3-1. `RuntimeOptions` — 인스턴스 설정

| 필드 | 타입 | 기본값 | 조건 |
| --- | --- | --- | --- |
| `model_dir` | `std::string` | 빈 문자열 | SD1.5 모델 디렉터리, 필수 |
| `tokenizer_dir` | `std::string` | 빈 문자열 | 비어 있으면 `<model_dir>/tokenizer` |
| `clip`, `unet`, `vae` | `ModelOptions` | High / None / false | 각 모델의 실행 설정 |
| `preview_decoder_path` | `std::string` | 빈 문자열 | TAESD decoder MNN 파일 (프리뷰 사용시) |
| `preview` | `ModelOptions` | High / None / false | 프리뷰 모델의 실행 설정 |

설정은 `Sd15Runtime` 생성 시 복사하고 인스턴스 수명 동안 유지한다. 이후 원본 설정 변경은 반영되지 않는다.

모델 디렉터리

```text
model-dir/
├── text_encoder.mnn
├── unet.mnn
├── vae_decoder.mnn
├── 필요한 .weight 파일
└── tokenizer/
    ├── vocab.json
    └── merges.txt
```

파일 규약

* 모델·tokenizer는 앱이 준비한 로컬 파일을 사용한다. 외부 가중치가 있으면 `<모델 경로>.weight`를 함께 배치한다.
* 변환기는 tokenizer를 만들지 않는다. SD1.5 CLIP의 `vocab.json`·`merges.txt`를 별도로 배포한다. `.mtok`와 다른 모델의 tokenizer는 지원하지 않는다.
* 모델 경로는 현재 ASCII 경로를 사용하며, tokenizer 파일은 각각 0바이트 초과~4MiB여야 한다.
* 실행 중 모델·tokenizer 파일을 변경하지 않는다.

### 3-2. `ModelOptions` — 모델별 실행 설정

| 필드 | 타입 | 기본값 | 허용값 |
| --- | --- | --- | --- |
| `precision` | `Precision` | `Precision::High` | `High`, `Normal`, `Low` |
| `tuning` | `Tuning` | `Tuning::None` | `None`, `Fast`, `Normal`, `Wide` |
| `low_memory` | `bool` | `false` | 압축 가중치 실행 경로 사용 여부 |

```cpp
sd15::RuntimeOptions config;
config.model_dir = "models/sd15";
config.clip.precision = sd15::Precision::Normal;
config.unet.tuning = sd15::Tuning::Fast;
config.unet.low_memory = true;
```

* 저장 방식은 OpenCL buffer로 고정한다. OpenCL 세션을 사용할 수 없으면 실패하며 CPU 세션 fallback은 거부한다.
* `precision`은 요청값이다. FP16 미지원 시 high로 전환될 수 있어 실제 실행 정밀도와 다를 수 있다.
* `low_memory`는 `MNN_LOW_MEMORY=ON` 빌드가 필요하다. Runtime configure에서 지원을 확인하지 못한 빌드는 이 요청을 거부한다.
* 디스크 캐시는 사용하지 않는다. 튜닝 비용은 생성 시간에 포함된다.

### 3-3. `GenerateOptions` — 요청별 설정

| 필드 | 타입 | 기본값 | 조건 |
| --- | --- | --- | --- |
| `prompt` | `std::string` | 빈 문자열 | UTF-8, 최대 16KiB, 빈 문자열 허용 |
| `negative_prompt` | `std::string` | 빈 문자열 | UTF-8, 최대 16KiB |
| `seed` | `int64_t` | -1 | 음수이면 요청마다 임의의 seed, 양수·0은 0~4294967295 |
| `steps` | `int` | 없음·필수 | 1~999 |
| `guidance` | `float` | 없음·필수 | 유한값 |
| `preview_interval` | `int` | 0 | 0이면 끔, 양수이면 해당 간격마다 프리뷰 |
| `sampling` | `SamplingPreset` | 없음·필수 | `DdimLeading`, `Dpmpp2mKarras` |

* `GenerateOptions(sampling, steps, guidance)`로 생성한다. 기본 생성자는 없으며 세 값은 호출자가 모델에 맞게 지정한다.

* `guidance`는 유한성만 검사하며 별도 범위 제한은 없다. `preview_interval`도 C++ API에서는 음수만 거부한다.
* prompt는 CLIP 기준 최대 77 tokens로 자르고 마지막 EOS를 보존한다. BOS=49406, EOS/padding=49407이다.
* NFC 정규화·Unicode 공백 병합·문자별 full lowercase를 적용한다. 비교 기준은 Transformers 5.17.0 / tokenizers 0.23.2의 Rust CLIP이며 ftfy를 사용하지 않는다.
* 소문자화는 root locale의 문자별 처리다. `İ`는 `i` + 결합점, `ΟΣ`는 `οσ`가 된다. Unicode 데이터 버전 차이까지 포함한 모든 입력의 일치는 보장하지 않는다.
* seed 재현은 동일 빌드·실행 환경 기준이다. 기존 NumPy seed 이미지와 Windows/Android 간 비트 일치는 보장하지 않는다.
* 학습 노이즈 설정은 scaled-linear/epsilon, train steps 1000, beta 0.00085~0.012다. VAE scale은 0.18215다.
* `DdimLeading`은 기존 leading, offset 1, eta 0, alpha-to-one false를 유지한다.
* `Dpmpp2mKarras`는 [Diffusers 0.40.0 구현](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)을 기준으로 한다. dpmsolver++, order 2, midpoint, Karras rho 7, final sigma 0, thresholding false이며 시작·마지막 스텝은 1차 갱신한다. 추가 난수는 사용하지 않는다.
* CLI는 `--sampling`, `--steps`, `--guidance`를 반드시 지정한다. 예: `--sampling dpmpp-2m-karras --steps 20 --guidance 7.5`. 수치는 사용 예시이며 기본값이 아니다.
* 기존 정수 timestep 모델을 사용한다. Euler/Euler a의 실수 linspace 시간표와 LCM 모델은 아직 지원하지 않는다.
* DPM 프리뷰는 현재 모델 출력에서 계산한 x0다. 최종 스텝은 이 x0를 VAE에 전달한다. Diffusers의 알고리즘을 기준으로 구현했으나, 부동소수점 연산 차이로 timestep과 최종 이미지가 달라질 수 있으며 비트 단위 일치를 보장하지 않는다.
* DPM 시간표는 초기화 시 계산한다. 반올림 경계에서는 부동소수점 차이로 Diffusers와 정수 timestep이 달라질 수 있으며, 최종 이미지에도 영향을 줄 수 있다. 특정 Python 환경의 시간표를 고정한 정적 표는 사용하지 않는다.

> prompt 16KiB·tokenizer 파일당 4MiB는 임시 방어 제한이며 모델 규격에 따른 값은 아니다.

## 4. 사용법·주의사항

### 4-1. 추론 실행

생성자는 파일 검사·tokenizer 로딩, `generate`는 모델 로딩·추론을 동기로 수행한다. 모두 UI 밖의 작업 스레드에서 호출한다. Runtime이 자체 백그라운드 스레드를 만드는 것은 아니다.

```cpp
sd15::RuntimeOptions config;
config.model_dir = "models/sd15";
sd15::Sd15Runtime runtime(config);

sd15::GenerateOptions request(sd15::SamplingPreset::Dpmpp2mKarras, 20, 7.5f);
request.prompt = "a photo of a cat";
const auto image = runtime.generate(request);
```

* 반환값의 `image.rgb`는 HWC 순서 RGB8이며 PNG 저장은 호출자가 담당한다. 임의 seed로 만든 이미지를 재현하려면 다음 요청에 `image.seed`를 지정한다.
* 한 인스턴스는 직렬 사용한다. 동시 호출·콜백 내 재진입은 `Runtime is busy` 예외다.
* `generate` 종료까지 Runtime 수명을 보장한다. 실행 중이나 콜백에서 파괴하지 않는다. 소멸자는 실행 중인 추론을 취소·대기하지 않는다.
* 정상·오류·취소 후 같은 인스턴스를 다시 사용할 수 있다. 생성자 성공이 이후 모델 실행 성공을 보장하지는 않는다.

### 4-2. 진행·프리뷰

```cpp
const auto image = runtime.generate(request, [](const std::string& stage, int step, int total) {
    // UI에 전달할 진행 정보는 복사해 큐에 넣는다.
});
```

* 진행·프리뷰 콜백은 `generate` 호출 스레드에서 실행한다. UI 갱신은 앱의 UI 큐로 전달하고 보관할 데이터는 콜백 안에서 복사한다.
* 진행 단계는 `text_encoder` → `unet` → `denoising` → `vae_decoder`다. 모델 로딩 전에는 `step=0`, denoising 완료 시에는 1부터 요청 스텝 수까지 전달한다.
* 프리뷰는 `preview_decoder_path`·양수 `preview_interval`·프리뷰 콜백을 함께 지정한다. 호출 형태는 `runtime.generate(request, progress, preview)`이며 프리뷰 인자는 `(int step, int total, const sd15::Image& image)`다.
* 프리뷰는 지정 간격에만 전달하며 마지막 스텝도 간격에 맞아야 한다. TAESD는 UNet과 함께 유지되므로 추가 메모리가 필요하다. 콜백에서 오래 처리하면 추론도 지연된다.
* 최종 이미지는 VAE로 만든다. 프리뷰와 다를 수 있으며 이후 오류·취소가 발생해도 이미 전달한 프리뷰는 회수하지 않는다.

### 4-3. 취소

```cpp
sd15::GenerationCancellationToken cancellation;
try {
    const auto image = runtime.generate(request, cancellation);
} catch (const sd15::GenerationCancelled&) {
    // 취소 완료
}
```

* 다른 스레드나 콜백에서 `cancellation.request_cancel()`을 호출한다. 토큰은 요청마다 새로 만들고 `generate`가 끝날 때까지 유지한다. 취소 오버로드에서도 진행·프리뷰 콜백을 뒤에 전달할 수 있다.
* `is_requested()`는 요청 여부다. 취소 완료는 `GenerationCancelled` 예외로 확인하며 모델 로딩·진행 중인 GPU 작업 때문에 지연될 수 있다.
* 모델 오류·콜백 예외는 그대로 전달한다. 마지막 취소 확인을 지난 요청은 성공할 수 있으며 실패·취소 시 최종 부분 이미지는 반환하지 않는다.
* 취소 요청 스레드에서 Runtime을 파괴하지 않는다. 취소 후 새 토큰으로 같은 Runtime에서 다시 추론할 수 있다.

## 5. 오류

입력·MNN·콜백 오류는 C++ 예외로 전달한다. Converter의 `ConvertError` 같은 결과 코드는 제공하지 않는다.

| 예외 | 대표 원인 |
| --- | --- |
| `GenerationCancelled` | 생성 취소 완료 |
| `std::invalid_argument` | 빈 모델 경로, 잘못된 모델 설정·생성 옵션·UTF-8, prompt 크기 초과, 프리뷰 필수 자원 누락 |
| `std::runtime_error` | busy, 모델 파일 누락, MNN 버전 불일치, OpenCL·모델 실행·텐서 계약·출력 유한성 오류 |
| 사용자 콜백 예외 | 진행·프리뷰 콜백에서 던진 예외를 그대로 전달 |
| 그 외 예외 | 파일 시스템·tokenizer JSON 파싱·메모리 할당 등의 오류 |

> 예외 표는 전체 목록이 아니다. tokenizer 자원과 파일 접근 오류는 하위 라이브러리의 예외가 그대로 전달될 수 있다.
