## 1. 모델 준비 및 의존성 관리

example:
```text
models/v1-5-pruned-emaonly.safetensors
```

`pyproject.toml`을 변경했다면 lockfile 갱신
```bash
uv lock
```

> 파일 수정시 이미지 재빌드 권장

## 2. ONNX export

```bash
docker build -f Dockerfile.export-onnx -t to-mnn:export-onnx .
```

```bash
docker run --rm \
  -e HF_HOME=/artifacts/huggingface \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:export-onnx
```

* 출력은 `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`와 필요한 외부 가중치
* 설정/tokenizer는 `artifacts/huggingface`에 캐시

> 재실행시 기존 ONNX 결과를 덮어씀

## 3. ONNX → MNN

```bash
docker build -f Dockerfile.export-mnn -t to-mnn:export-mnn .
```

```bash
docker run --rm --network none \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:export-mnn
```

* 입력: `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`
* 출력: `artifacts/mnn/{text_encoder,unet,vae_decoder}.mnn`
* 변환 로그: `artifacts/mnn/{모델명}.convert.log`
* `unet.mnn.weight`가 생성되면 `unet.mnn`과 함께 보관
* `--fp16` 가중치 저장 옵션으로 변환하며, 실패하면 중단

> 재실행시 같은 이름의 모델과 로그를 덮어씀

## 4. 추론 이미지 빌드 및 이미지 생성

```bash
docker build -f Dockerfile.runtime -t to-mnn:runtime .
mkdir -p img
```

기존 설정/tokenizer 캐시와 MNN 모델을 사용해 네트워크 없이 생성합니다.

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/artifacts:/artifacts:ro" \
  -v "$(pwd -W)/img:/img" \
  to-mnn:runtime
```

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/artifacts:/artifacts:ro" \
  -v "$(pwd -W)/img:/img" \
  to-mnn:runtime \
  --prompt "a photo of a cat sitting on a wooden table, natural light" \
  --steps 20 \
  --seed 42 \
  --guidance 7.5 \
  --threads 4 \
  --output /img/cat_seed42.png
```

> seed 미지정시 랜덤

## 5. 템플릿 생성 (개발 PC)

* 앞 단계의 safetensors, ONNX, MNN을 사용
* `to-mnn:export-onnx` 이미지가 필요
* 출력: `artifacts/templates/sd15/{graph.bin,manifest.json}`

```bash
docker build -f Dockerfile.gen-template -t to-mnn:gen-template .
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:gen-template
```

> `graph.bin`은 고정된 그래프 데이터고, `manifest.json`은 가중치를 어디에 어떤 형식으로 넣을지 정의한 파일

## 6. C++ 스트리밍 변환

```bash
docker build -f Dockerfile.cpp.sd15 -t to-mnn:cpp-sd15 .
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  --memory=64m --memory-swap=64m \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:cpp-sd15 \
  --checkpoint /models/v1-5-pruned-emaonly.safetensors \
  --template-dir /artifacts/templates/sd15 \
  --output /artifacts/mnn-cpp
```

> `--chunk-bytes`로 입력 버퍼를 조정할 수 있음

## 7. 변환 결과를 텐서 단위로 비교

* 기준 MNN과 C++로 복원한 MNN의 텐서 이름, shape, dtype, 저장값과 그래프 바이트가 일치하는지 비교
* 결과는 `tensor-comparison.json`
```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  --entrypoint python \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:gen-template -m src.sd15.compare_mnn \
  --reference /artifacts/mnn \
  --candidate /artifacts/mnn-cpp \
  --manifest /artifacts/templates/sd15/manifest.json
```

* 작은 테스트 데이터를 만들어 변환기, 템플릿, 생성기, 비교기 코드가 정상 동작하는지 검사
* 잘못된 입력을 거부하는지 확인하며 실제 모델 파일은 불필요
```bash
docker build -f Dockerfile.cpp.sd15 -t to-mnn:cpp-sd15 .
docker build -f Dockerfile.gen-template -t to-mnn:gen-template .
docker build -f Dockerfile.test.sd15 -t to-mnn:test-sd15 .
docker run --rm --network none to-mnn:test-sd15
```

# Windows Native OpenCL Runtime Infer

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
scripts/build-mnn.sh --python
uv run -m src.sd15.infer_mnn --backend OPENCL
```

> output: `img/mnn_sd15.png`


# 코드 검사 및 포맷

```bash
uv run --locked --only-dev ruff check .
uv run --locked --only-dev ruff format --check .
```

```bash
uv run --locked --only-dev ruff check --fix .
uv run --locked --only-dev ruff format .
```
