# SD1.5 Runtime

## 1. 현재 범위

* C++17 정적 라이브러리: `sd15_runtime`, CLI: `sd15-infer`
* Windows 네이티브 MNN 3.6.1, OpenCL high precision / buffer / tuning none
* SD1.5, batch 1, 512×512, CFG + DDIM eta=0
* CLIP → UNet → VAE를 순서대로 로딩·실행·해제
* OpenCL 세션 선택을 확인하고 CPU 세션으로의 fallback은 거부. 개별 연산의 CPU 보조 실행까지 금지하는 의미는 아님

> 첫 목표는 C++ 이미지 생성. 토큰화·초기 노이즈는 Python에서 준비하며, 네이티브 토크나이저·취소·비동기 작업·모델 캐시는 미구현

## 2. Windows 빌드

* 기존 [MNN 빌드](../README.md#3-mnnruntime-빌드)를 먼저 실행
* MSVC x64, Windows SDK, CMake, Ninja 필요
* 프로젝트 루트의 Windows Git Bash에서 실행

```bash
bash scripts/build-runtime.sh
```

* 기존 `MNNRuntime/build/{MNN.dll,MNN.lib}`에 연결. MNN 자체는 재빌드하지 않음
* Converter는 끄고 Runtime·CLI·C++ 테스트만 빌드
* 출력: `build/windows-runtime/cli/sd15-infer.exe`
* 실행 파일 옆에 `MNN.dll` 복사
* C++ 테스트 실패 시 빌드 스크립트 실패
* 빌드 병렬 수: `BUILD_JOBS=4 bash scripts/build-runtime.sh` (기본 8)

## 3. 입력 준비와 추론

* 모델: 기존 Docker 흐름으로 생성한 `artifacts/mnn-cpp-hqq-b128-manual`
* tokenizer/config: ONNX export가 만든 `artifacts/huggingface`
* 입력 준비는 기존 Python runner와 같은 tokenizer·NumPy 난수를 사용
* `uv sync --locked --group runtime`으로 Python 환경 준비

```bash
uv run -m src.sd15.prepare_inference \
  --prompt "a photo of a cat sitting on a wooden table, natural light" \
  --seed 42 --steps 20 --guidance 7.5 \
  --output artifacts/runtime-cat-input.json

./build/windows-runtime/cli/sd15-infer.exe \
  --model-dir artifacts/mnn-cpp-hqq-b128-manual \
  --input artifacts/runtime-cat-input.json \
  --output img/cat-cpp-sd15.png
```

* JSON: 양·음 프롬프트의 token IDs 각 77개, F32 초기 latent 16,384개, 생성 옵션·설정·seed
* JSON은 검증용 입력 묶음. MNN 모델이나 Converter manifest 형식과 별개
* CLI는 JSON 최대 2MiB, 고정 SD1.5 scaled-linear/epsilon 스케줄만 지원
* 단계별 진행과 전체 시간을 출력하고 RGB PNG 저장. 기존 출력 파일은 거부
* 준비된 JSON 이후에는 Python 없이 C++ CLI만 실행 가능

## 4. C++ API

헤더: [runtime.hpp](../cpp/runtime/include/sd15/runtime.hpp)

```cpp
sd15::GenerateOptions options;
// model_dir, tokens, negative_tokens, initial_latent를 준비
const auto image = sd15::generate(options, [](const std::string& stage, int step, int total) {
    // 호출 스레드에서 진행 상태 수신
});
```

* 동기 API. 입력·MNN·콜백 오류는 C++ 예외로 전달하며 CLI 경계에서 처리
* 결과: `width`, `height`, HWC 순서의 RGB8 `rgb`
* PNG 저장은 CLI 담당. 라이브러리는 Python·JSON·이미지 인코더에 의존하지 않음
* MNN 입출력 name·shape·dtype와 결과의 NaN/Inf 검사
* UNet 실행마다 sample·timestep·context를 모두 다시 복사. 실행 후 입력 버퍼 보존을 가정하지 않음
* CLI의 JSON·PNG 구현은 고정 MNN 소스에 포함된 헤더를 사용
