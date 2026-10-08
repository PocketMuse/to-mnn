## 1. 모델 준비 및 의존성 관리

* 아래 명령은 프로젝트 루트의 Windows Git Bash에서 실행
* UNet: W8 + HQQ + Block128, CLIP/VAE: FP16
* Docker Compose 2.22 이상이 필요하며 OpenCL 추론은 Windows 네이티브 환경에서 실행

모델 경로:

```text
models/v1-5-pruned-emaonly.safetensors
```

`pyproject.toml`을 변경했다면 lockfile 갱신

```bash
uv lock
```

## 2. SD1.5 FP32 → C++ W8 양자화

```bash
docker compose -f compose.sd15.yaml build
docker compose -f compose.sd15.yaml run --rm export-onnx
docker compose -f compose.sd15.yaml run --rm reference-fp16
docker compose -f compose.sd15.yaml run --rm reference-w8
docker compose -f compose.sd15.yaml run --rm template
docker compose -f compose.sd15.yaml run --rm convert
docker compose -f compose.sd15.yaml run --rm compare
```

### `build` — SD1.5 이미지 빌드

* 실행 설정: `compose.sd15.yaml`, Dockerfile: `docker/sd15/`

> 소스나 의존성을 변경했다면 이미지를 다시 빌드

### `export-onnx` — ONNX 생성

* 출력은 `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`와 필요한 외부 가중치
* 설정/tokenizer는 `artifacts/huggingface`에 캐시
* Windows 추론에서도 읽을 수 있도록 캐시의 Linux 파일 링크를 일반 파일로 자동 교체

> 재실행시 기존 ONNX 결과를 덮어씀

### `reference-fp16` — FP16 참조 MNN 준비

* W8 템플릿의 원본 가중치 매핑과 CLIP/VAE 준비에 필요
* 입력: `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`
* 출력: `artifacts/mnn/{text_encoder,unet,vae_decoder}.mnn`
* 변환 로그: `artifacts/mnn/{모델명}.convert.log`
* `unet.mnn.weight`가 생성되면 `unet.mnn`과 함께 보관
* 변환 설정 고정: `--fp16 --transformerFuse --optimizePrefer 2 --optimizeLevel 1`

> 재실행시 같은 이름의 모델과 로그를 덮어씀

### `reference-w8` — UNet W8 기준 모델 준비

* `reference-fp16`의 CLIP/VAE를 복사하고 UNet을 W8 + HQQ + Block128로 변환
* UNet은 `--fp16` 없이 변환하여 상수의 원래 정밀도를 유지
* 변환 성공·HQQ 활성화 로그와 출력 파일을 자동 확인
* 변환 로그: `artifacts/quant-revalidation/w8-block128-hqq/unet.convert.log`
* 입력·출력 경로를 바꾸려면 `compose.sd15.yaml`의 해당 서비스 설정을 수정
* 출력: `artifacts/quant-revalidation/w8-block128-hqq/{text_encoder,unet,vae_decoder}.mnn`

> 재실행시 같은 이름의 모델을 덮어씀

### `template` — W8 템플릿 생성 (개발 PC)

* 앞 단계에서 만든 ONNX, FP16 참조 MNN, W8 기준 모델을 사용
* 출력: `artifacts/templates/sd15-hqq-b128-manual/{graph.bin,manifest.json}`
* `graph.bin`은 고정 그래프 데이터, `manifest.json`은 가중치 위치·저장 형식

> 기존 출력 폴더는 덮어쓰지 않음. 재실행시 Compose의 출력 경로와 이후 단계의 참조 경로를 함께 변경

### `convert` — C++ W8 스트리밍 변환

* MNNConvert나 MNN Runtime 없이 safetensors와 템플릿으로 변환
* 출력: `artifacts/mnn-cpp-hqq-b128-manual/{text_encoder,unet,vae_decoder}.mnn`
* `convert` 서비스의 `command`에 `--chunk-bytes`를 추가해 입력 버퍼 조정 가능 (기본 1MiB)
* 64MiB 제한은 Docker 검증 조건이며 Android 검증 결과는 아님

> 기존 출력 폴더는 덮어쓰지 않음. 재실행시 Compose의 출력·비교 경로와 4번의 추론 경로를 함께 변경

### `compare` — 텐서 단위 비교

* `reference-w8`의 기준 모델과 `convert`의 C++ 출력 모델을 비교
* 텐서 이름·shape·dtype, W8 저장값·FP32 min/scale, 비양자화 텐서와 고정 그래프 바이트 확인
* 결과: `artifacts/tensor-comparison-hqq-b128-manual.json`
* 모든 텐서가 일치하고 `graph=True`인지 확인. 불일치시 종료 코드 1

> `template`, `convert`, `compare`는 GPU 없이 실행. OpenCL 추론과 모바일 수치 재현성은 별도 검증

## 3. MNNRuntime 빌드

Requires:

* Git Bash
* MSVC x64
* Windows SDK
* CMake
* Ninja
* uv
* Python 3.12
* OpenCL 지원 GPU 드라이버

```bash
uv sync --locked --group runtime
bash scripts/build-mnn.sh --python
```

* MNN 3.6.1 소스를 내려받아 OpenCL Runtime과 Python 바인딩을 빌드
* `--python`은 4번의 Python 추론 실행에 필요
* Runtime 출력: `MNNRuntime/build/MNN.dll`
* Python 바인딩: `MNNRuntime/python/_mnncengine*.pyd`
* 빌드 로그: `MNNRuntime/{configure,build,python-configure,python-build}.log`

## 4. C++ W8 모델로 이미지 추론

* 3번의 Windows OpenCL Runtime과 2번에서 생성한 C++ 출력 모델을 사용

```bash
uv run -m src.sd15.infer_mnn \
  --model-dir artifacts/mnn-cpp-hqq-b128-manual \
  --prompt "a photo of a cat sitting on a wooden table, natural light" \
  --steps 20 \
  --guidance 7.5 \
  --output img/cat-hqq-cpp.png
```

> seed 미지정시 랜덤

## 테스트

* 작은 테스트 데이터로 변환기·템플릿 생성기·비교기와 잘못된 입력 처리를 검사
* 실제 모델 파일은 불필요

```bash
docker compose -f compose.sd15.yaml run --rm test
```

## 코드 검사 및 포맷

```bash
uv run --locked --only-dev ruff check .
uv run --locked --only-dev ruff format --check .
```

```bash
uv run --locked --only-dev ruff check --fix .
uv run --locked --only-dev ruff format .
```
