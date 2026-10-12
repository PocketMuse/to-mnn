# SD1.5 Runtime

## 1. 현재 범위

* C++17 정적 라이브러리 `sd15_runtime`, 동기 클래스 `Sd15Runtime`, CLI `sd15-infer`
* prompt·negative prompt·seed → C++ 토큰화·초기 노이즈 → CLIP → UNet → VAE → RGB8
* Windows 네이티브 MNN 3.6.1, OpenCL high precision / buffer / tuning none
* SD1.5, batch 1, 512×512, CFG + DDIM eta=0. steps 1~999, 기본 20, guidance 기본 7.5
* scaled-linear/epsilon, train steps 1000, beta 0.00085~0.012, offset 1, alpha-to-one false, VAE scale 0.18215 고정
* 필요한 모델만 순서대로 로딩·실행·해제. 생성 간 무거운 모델은 보관하지 않음
* OpenCL 세션 선택을 확인하고 CPU 세션 fallback은 거부. 개별 연산의 CPU 보조 실행까지 금지하는 의미는 아님

> Python은 모델 준비·비교 도구에서만 사용.

## 2. Windows 빌드

* 기존 [MNN 빌드](../README.md#3-mnnruntime-빌드)를 먼저 실행
* MSVC x64, Windows SDK, CMake, Ninja 필요. 프로젝트 루트의 Git Bash에서 실행

```bash
bash scripts/build-runtime.sh
```

* 기존 `MNNRuntime/build/{MNN.dll,MNN.lib}`에 연결. MNN 자체는 재빌드하지 않음
* 고정 MNN 소스의 Unicode 처리와 JSON 헤더 사용. LLM·공식 diffusion 엔진은 빌드하지 않음
* Converter는 끄고 Runtime·CLI·C++ 테스트만 빌드
* 출력: `build/windows-runtime/cli/sd15-infer.exe`, 실행 파일 옆에 `MNN.dll` 복사
* C++ 테스트 실패 시 스크립트 실패. `BUILD_JOBS=4`로 빌드 병렬 수 조정 가능

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
* Windows CLI는 Unicode 명령행을 UTF-8 prompt로 변환. MNN 모델 경로는 현재 ASCII 경로 사용

## 4. C++ API와 수명

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
* UTF-8 검사, Unicode 소문자화·공백 처리. 각 prompt는 최대 16KiB
* NFC 조합 정규화·ftfy 보정·여러 문자로 확장되는 소문자 변환은 미지원. 모든 Unicode 입력의 Python 토큰 일치를 보장하지 않음
* 앱이 배포한 SD1.5 CLIP 어휘 사용. `.mtok`와 다른 모델의 tokenizer는 지원하지 않음
* C++ `mt19937` + `normal_distribution<float>(0, 1)`로 float 16,384개 생성
* 동일 빌드·실행 환경의 고정 seed 재현을 검사. 기존 NumPy seed 이미지 및 Windows/Android 간 비트 일치는 보장하지 않음

> prompt 16KiB·tokenizer 파일당 4MiB·내부 scheduler 검증의 학습 timestep 10,000 상한은 모델 규격이나 실측에 따른 값이 아닌 임시 방어 제한이며, 추후 필요에 따라 조정 가능.

공식 참고:

* [MNN 3.6.1 StableDiffusion](https://github.com/alibaba/MNN/blob/3.6.1/transformers/diffusion/engine/src/stable_diffusion.cpp): 클래스 수명·난수·모델 순차 해제
* [MNN tokenizer](https://github.com/alibaba/MNN/blob/3.6.1/transformers/diffusion/engine/src/tokenizer.cpp): 공식 wrapper의 BOS/EOS·0 padding. 현재 CLIP 계약에 맞게 EOS padding 사용
* [HuggingFace CLIP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py): byte BPE·`end_of_word_suffix`·특수 토큰 규칙
