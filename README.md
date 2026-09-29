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
