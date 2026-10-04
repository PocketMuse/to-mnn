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
* Windows 추론에서도 읽을 수 있도록 캐시의 Linux 파일 링크를 일반 파일로 자동 교체
* UNet의 마스크 없는 Attention은 `bmm × scale`로 export하여 대형 0 상수 생성을 방지
* 기존 export 결과에는 이 수정이 적용되지 않으므로 ONNX → MNN → 템플릿을 다시 생성해야 함

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
* 변환 설정 고정: `--fp16 --transformerFuse --optimizePrefer 2 --optimizeLevel 1`

> 재실행시 같은 이름의 모델과 로그를 덮어씀

## 4. MNNRuntime 빌드

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
* `--python`은 5번의 Python 추론 실행에 필요
* Runtime 출력: `MNNRuntime/build/MNN.dll`
* Python 바인딩: `MNNRuntime/python/_mnncengine*.pyd`
* 빌드 로그: `MNNRuntime/{configure,build,python-configure,python-build}.log`

## 5. 이미지 추론

```bash
uv run -m src.sd15.infer_mnn \
  --prompt "a photo of a cat sitting on a wooden table, natural light" \
  --steps 20 \
  --guidance 7.5 \
  --output img/cat.png
```

> seed 미지정시 랜덤

## 6. 템플릿 생성 (개발 PC)

* 앞 단계의 safetensors, ONNX, MNN을 사용
* `to-mnn:export-onnx` 이미지가 필요
* 출력: `artifacts/templates/sd15/{graph.bin,manifest.json}`
* `--transformerFuse`로 생성된 비양자화 `AttentionParam`은 고정 그래프 바이트로 보존
* 템플릿 생성·복원·바이트 비교는 GPU 없이 실행하며, OpenCL 추론 검증과는 별개

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


# 코드 검사 및 포맷

```bash
uv run --locked --only-dev ruff check .
uv run --locked --only-dev ruff format --check .
```

```bash
uv run --locked --only-dev ruff check --fix .
uv run --locked --only-dev ruff format .
```
