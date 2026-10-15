# SD1.5 Runtime

## 1. 현재 범위

* C++17 정적 라이브러리 `sd15_runtime`, 동기 클래스 `Sd15Runtime`, CLI `sd15-infer`
* prompt·negative prompt·seed → C++ 토큰화·초기 노이즈 → CLIP → UNet → VAE → RGB8
* Windows 네이티브 MNN 3.6.1, 기본 OpenCL high precision / buffer / tuning none
* SD1.5, batch 1, 512×512, CFG + DDIM eta=0. steps 1~999, 기본 20, guidance 기본 7.5
* scaled-linear/epsilon, train steps 1000, beta 0.00085~0.012, offset 1, alpha-to-one false, VAE scale 0.18215 고정
* 필요한 모델만 순서대로 로딩·실행·해제. 생성 간 무거운 모델은 보관하지 않음
* OpenCL 세션 선택을 확인하고 CPU 세션 fallback은 거부. 개별 연산의 CPU 보조 실행까지 금지하는 의미는 아님

> Python은 모델 준비·비교 도구에서만 사용.

## 2. Windows 빌드

* 기존 [MNN 빌드](../README.md#3-mnnruntime-빌드)를 먼저 실행
* MSVC x64, Windows SDK, CMake, Ninja 필요. 프로젝트 루트의 Git Bash에서 실행
* 설치된 vcpkg의 `VCPKG_ROOT` 환경변수 필요. JSON·ICU·CLI11·테스트용 Catch2는 configure 시 manifest 기준으로 준비
* vcpkg 도입 전의 빌드 폴더는 기존 CMake 캐시를 초기화한 뒤 실행

```bash
bash scripts/build-runtime.sh
```

* 기존 `MNNRuntime/build/{MNN.dll,MNN.lib}`에 연결. MNN 자체는 재빌드하지 않음
* Unicode 정규화·문자 분류·소문자화·정규식은 vcpkg의 ICU 사용. MNN Unicode 보조 소스는 더 이상 우리 Runtime에 컴파일하지 않음
* Converter는 끄고 Runtime·CLI·C++ 테스트만 빌드
* 출력: `build/windows-runtime/cli/sd15-infer.exe`, 실행 파일 옆에 `MNN.dll`과 vcpkg의 ICU DLL 배치. 실행 파일 배포 시 이 DLL들도 포함
* C++ 테스트 실패 시 스크립트 실패. `BUILD_JOBS=4`로 빌드 병렬 수 조정 가능

선택적 통합 테스트는 `SD15_TEST_TOKENIZER_DIR`, `SD15_TEST_MODEL_DIR` 환경변수로 빌드 스크립트에 자원 경로를 전달한다. 모델 검증은 Windows 네이티브 OpenCL에서 실행한다.

```bash
# 이미 빌드한 테스트를 직접 실행
./build/windows-runtime/runtime/sd15-tokenizer-test.exe --tokenizer-dir TOKENIZER_DIR
./build/windows-runtime/runtime/sd15-generation-test.exe --model-dir MODEL_DIR --tokenizer-dir TOKENIZER_DIR
```

토크나이저 대조 도구는 `--tokenizer-dir DIR --stdin`으로 UTF-8 prompt를 읽고 token IDs만 출력한다.

## 3. 모델 자원과 실행

모델과 함께 CLIP의 `vocab.json`, `merges.txt`를 배포한다. 변환기는 모델 파일만 만들며 tokenizer는 모델 계열의 공통 자원이다.

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

기존 ONNX export 캐시에서 tokenizer 파일 복사(Git Bash, 개발 PC에서 한 번):

```bash
cache=artifacts/huggingface/models--stable-diffusion-v1-5--stable-diffusion-v1-5
revision=$(cat "$cache/refs/main")
mkdir -p artifacts/mnn-cpp-hqq-b128-manual/tokenizer
cp "$cache/snapshots/$revision/tokenizer/"{vocab.json,merges.txt} \
  artifacts/mnn-cpp-hqq-b128-manual/tokenizer/

./build/windows-runtime/cli/sd15-infer.exe \
  --model-dir artifacts/mnn-cpp-hqq-b128-manual \
  --prompt "a photo of a cat sitting on a wooden table, natural light" \
  --steps 20 --guidance 7.5 \
  --output img/cat-native-runtime.png
```

* 별도 tokenizer 위치는 `--tokenizer-dir DIR`, 음 프롬프트는 `--negative-prompt TEXT`
* seed는 signed 64-bit 정수, 기본 -1. 음수이면 요청마다 임의의 seed를 선택하고, 0~4294967295이면 그대로 사용. 그보다 큰 양수는 거부
* 단계별 진행·전체 시간·seed를 출력하고 PNG 저장. 기존 출력 파일은 거부
* prompt·seed를 직접 전달하며, 사전 생성한 token IDs·latent JSON은 사용하지 않음
* CLI11로 옵션을 파싱하며 `--help`로 사용법 확인. Windows Unicode 명령행의 UTF-8 변환도 CLI11이 처리. MNN 모델 경로는 현재 ASCII 경로 사용

## 4. C++ API와 수명

실행 옵션은 `RuntimeOptions`에서 지정하며 인스턴스 생성 후 고정된다.
내부에서는 모델별 공통 `ModelOptions`로 변환한다. `ClipEncoder`·`UNetDenoiser`·`VaeDecoder`가 입출력 계약을, `MnnModelSession`이 MNN 설정과 세션을 담당한다. 이 클래스들은 내부 구현이며 공개 API는 아래와 같다.

| 필드 | 값 / 기본값 | CLI |
| --- | --- | --- |
| `unet_precision` | `Precision::High`(기본), `Normal`, `Low` | `--unet-precision high\|normal\|low` |
| `tuning` | `Tuning::None`(기본), `Fast`, `Normal`, `Wide` | `--tuning none\|fast\|normal\|wide` |
| `unet_low_memory` | `false`(기본), `true` | `--unet-low-memory` |

* 정밀도·low-memory는 UNet에만 적용. CLIP/VAE는 high/normal-memory 유지. 튜닝은 세 모델 모두 적용하며 저장 방식은 buffer 고정.
* OpenCL 기본 커널 설정은 High=FP32 저장·계산, Normal=FP16 저장·FP32 계산, Low=FP16 저장·계산. 개별 커널의 일부 계산은 다를 수 있다.
* FP16 미지원 GPU에서는 MNN이 high로 전환한다.
* low-memory는 압축 가중치 실행 경로이며 네이티브 INT8 연산을 보장하지 않는다.
* `build-mnn.sh`는 `MNN_LOW_MEMORY=ON`으로 빌드한다. Runtime configure는 `MNN_RUNTIME_DIR/CMakeCache.txt`에서 지원 여부를 확인하며, ON을 확인하지 못한 빌드는 low-memory 요청을 거부한다. 엔진 재빌드 후 Runtime도 다시 configure/build한다.
* 디스크 캐시는 사용하지 않는다. 튜닝 비용이 생성 시간에 포함되므로 전체 시간과 반복 denoising 시간을 구분한다.

```cpp
sd15::RuntimeOptions config;
config.model_dir = "models/sd15";
config.unet_precision = sd15::Precision::Normal;
config.tuning = sd15::Tuning::Fast;
config.unet_low_memory = true;
```

헤더: [runtime.hpp](../cpp/runtime/include/sd15/runtime.hpp)

```cpp
sd15::RuntimeOptions config;
config.model_dir = "models/sd15";
sd15::Sd15Runtime runtime(config);

sd15::GenerateOptions request;
request.prompt = "a photo of a cat";
request.negative_prompt = "blurry";
request.seed = -1; // 생략해도 랜덤. 재현할 때는 image.seed를 지정
request.steps = 20;
request.guidance = 7.5f;
const auto image = runtime.generate(request, [](const std::string& stage, int step, int total) {
    // 호출 스레드에서 진행 상태 수신
});
```

* 생성자는 경로를 검사하고 작은 tokenizer 자원만 읽음. CLIP·UNet·VAE는 `generate`의 해당 단계에서 로딩
* Runtime 경로 설정은 생성 시 고정. 생성 요청은 실행 시작 시 복사하고 호출마다 RNG·latent·embedding을 새로 생성
* 한 인스턴스의 동시·콜백 내 재진입 생성은 `Runtime is busy` 예외. 앱은 Runtime 하나를 직렬 사용
* CLIP 해제 후에도 embedding, UNet 해제 후에도 latent는 독립 host 버퍼에 보존
* 성공·모델 오류·콜백 예외 모두 RAII로 모델 자원을 해제. 예외 후 같은 인스턴스에서 다시 생성 가능
* 생성 중 객체 파괴·콜백에서 객체 파괴는 금지. 호출자는 완료까지 Runtime 수명을 보장
* 진행 단계는 `text_encoder`, `unet`, `denoising`, `vae_decoder`. 로딩 단계에서는 step=0
* 결과는 실제 사용한 `seed`, `width`, `height`, HWC 순서 RGB8 `rgb`. 재현하려면 결과의 `seed`를 다음 요청에 지정. PNG 저장은 CLI 담당
* 입력·MNN·콜백 오류는 C++ 예외로 전달. MNN 입출력 name·shape·dtype와 출력 NaN/Inf 검사
* UNet 실행마다 sample·timestep·context를 다시 복사. 입력 버퍼 보존을 가정하지 않음

> 자원 객체의 해제와 OS·GPU 드라이버의 즉시 메모리 반환은 다름. PC 결과를 Android 메모리 검증으로 간주하지 않음

## 5. 토큰화와 난수

* CLIP byte BPE, 단어 끝 `</w>`, BOS 49406·EOS/padding 49407, 최대 77 tokens. 잘릴 때 마지막 EOS 보존
* 잘못된 UTF-8은 거부. 각 prompt는 최대 16KiB
* ICU로 NFC → Unicode 공백 병합 → 문자별 full lowercase → CLIP 정규식 분할을 수행
* 비교 기준은 Transformers 5.17.0 / tokenizers 0.23.2의 Rust 기반 CLIP. 이 경로는 ftfy를 사용하지 않음
* 소문자화는 root locale에서 문자별로 수행. `İ`는 `i` + 결합점으로 확장하며 `ΟΣ`는 `οσ`가 된다. 문자열 전체 소문자화나 case folding으로 대체하지 않음
* 분해형 악센트·한글 자모·결합 문자·Unicode 공백 등의 기준 token IDs를 회귀 테스트로 고정. Unicode 데이터 버전 차이까지 포함한 모든 입력의 일치를 보장하지는 않음
* 앱이 배포한 SD1.5 CLIP 어휘 사용. `.mtok`와 다른 모델의 tokenizer는 지원하지 않음
* C++ `mt19937` + `normal_distribution<float>(0, 1)`로 float 16,384개 생성
* 동일 빌드·실행 환경의 고정 seed 재현을 검사. 기존 NumPy seed 이미지 및 Windows/Android 간 비트 일치는 보장하지 않음

> prompt 16KiB·tokenizer 파일당 4MiB·내부 scheduler 검증의 학습 timestep 10,000 상한은 모델 규격이나 실측에 따른 값이 아닌 임시 방어 제한이며, 추후 필요에 따라 조정 가능.

공식 참고:

* [MNN 3.6.1 StableDiffusion](https://github.com/alibaba/MNN/blob/3.6.1/transformers/diffusion/engine/src/stable_diffusion.cpp): 클래스 수명·난수·모델 순차 해제
* [MNN tokenizer](https://github.com/alibaba/MNN/blob/3.6.1/transformers/diffusion/engine/src/tokenizer.cpp): 공식 wrapper의 BOS/EOS·0 padding. 현재 CLIP 계약에 맞게 EOS padding 사용
* [HuggingFace CLIP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py): byte BPE·`end_of_word_suffix`·특수 토큰 규칙
