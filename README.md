## 1. 모델 준비 및 의존성 관리

* 아래 명령은 Windows Git Bash에서 실행
* UNet: W8 + HQQ + Block128, CLIP/VAE: FP16
* Docker가 필요하며 OpenCL 추론은 Windows 네이티브 환경에서 실행

모델 경로:

```text
models/v1-5-pruned-emaonly.safetensors
```

`pyproject.toml`을 변경했다면 lockfile 갱신

```bash
uv lock
```

> 소스나 의존성을 변경했다면 해당 Docker 이미지를 다시 빌드

## 2. ONNX export

```bash
docker build -f Dockerfile.export-onnx -t to-mnn:export-onnx .
```

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -e HF_HOME=/artifacts/huggingface \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:export-onnx
```

* 출력은 `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`와 필요한 외부 가중치
* 설정/tokenizer는 `artifacts/huggingface`에 캐시
* Windows 추론에서도 읽을 수 있도록 캐시의 Linux 파일 링크를 일반 파일로 자동 교체

> 재실행시 기존 ONNX 결과를 덮어씀

## 3. FP16 참조 MNN 준비

* W8 템플릿의 원본 가중치 매핑과 CLIP/VAE 준비에 필요

```bash
docker build -f Dockerfile.export-mnn -t to-mnn:export-mnn .
```

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:export-mnn
```

* 입력: `artifacts/onnx/{text_encoder,unet,vae_decoder}/model.onnx`
* 출력: `artifacts/mnn/{text_encoder,unet,vae_decoder}.mnn`
* 변환 로그: `artifacts/mnn/{모델명}.convert.log`
* `unet.mnn.weight`가 생성되면 `unet.mnn`과 함께 보관
* 변환 설정 고정: `--fp16 --transformerFuse --optimizePrefer 2 --optimizeLevel 1`

> 재실행시 같은 이름의 모델과 로그를 덮어씀

## 4. UNet W8 기준 모델 준비

* 3번의 CLIP/VAE를 복사하고 UNet을 W8 + HQQ + Block128로 변환
* UNet은 `--fp16` 없이 변환하여 상수의 원래 정밀도를 유지

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:export-mnn \
  --unet-quantization hqq-b128 \
  --output-dir /artifacts/quant-revalidation/w8-block128-hqq
```

* 변환 성공·HQQ 활성화 로그와 출력 파일을 자동 확인
* 변환 로그: `artifacts/quant-revalidation/w8-block128-hqq/unet.convert.log`
* FP16 참조 경로를 바꿨다면 `--reference-mnn-dir`로 지정
* 출력: `artifacts/quant-revalidation/w8-block128-hqq/{text_encoder,unet,vae_decoder}.mnn`

> 재실행시 같은 이름의 모델을 덮어씀

## 5. W8 템플릿 생성 (개발 PC)

```bash
docker build -f Dockerfile.gen-template -t to-mnn:gen-template .
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:gen-template \
  --mnn-dir /artifacts/quant-revalidation/w8-block128-hqq \
  --reference-mnn-dir /artifacts/mnn \
  --unet-quantization hqq-b128 \
  --output /artifacts/templates/sd15-hqq-b128-manual
```

* 2번의 ONNX, 3번의 FP16 참조 MNN, 4번의 W8 기준 모델을 사용
* 출력: `artifacts/templates/sd15-hqq-b128-manual/{graph.bin,manifest.json}`
* `graph.bin`은 고정 그래프 데이터, `manifest.json`은 가중치 위치·저장 형식

> 기존 출력 폴더는 덮어쓰지 않음. 재실행시 새 경로를 지정하고 이후 명령에도 반영

## 6. C++ W8 스트리밍 변환

```bash
docker build -f Dockerfile.cpp.sd15 -t to-mnn:cpp-sd15 .
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  --memory=64m --memory-swap=64m \
  -v "$(pwd -W)/models:/models:ro" \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:cpp-sd15 \
  --checkpoint /models/v1-5-pruned-emaonly.safetensors \
  --template-dir /artifacts/templates/sd15-hqq-b128-manual \
  --output /artifacts/mnn-cpp-hqq-b128-manual
```

* MNNConvert나 MNN Runtime 없이 safetensors와 템플릿으로 변환
* 출력: `artifacts/mnn-cpp-hqq-b128-manual/{text_encoder,unet,vae_decoder}.mnn`
* `--chunk-bytes`로 입력 버퍼 조정 가능 (기본 1MiB)
* 64MiB 제한은 Docker 검증 조건이며 Android 검증 결과는 아님

> 기존 출력 폴더는 덮어쓰지 않음. 재실행시 새 경로를 지정하고 이후 명령에도 반영

## 7. 변환 결과를 텐서 단위로 비교

```bash
MSYS_NO_PATHCONV=1 docker run --rm --network none \
  --entrypoint python \
  -v "$(pwd -W)/artifacts:/artifacts" \
  to-mnn:gen-template -m src.sd15.compare_mnn \
  --reference /artifacts/quant-revalidation/w8-block128-hqq \
  --candidate /artifacts/mnn-cpp-hqq-b128-manual \
  --manifest /artifacts/templates/sd15-hqq-b128-manual/manifest.json \
  --report /artifacts/tensor-comparison-hqq-b128-manual.json
```

* 4번의 기준 모델과 6번의 C++ 출력 모델을 비교
* 텐서 이름·shape·dtype, W8 저장값·FP32 min/scale, 비양자화 텐서와 고정 그래프 바이트 확인
* 결과: `artifacts/tensor-comparison-hqq-b128-manual.json`
* 모든 텐서가 일치하고 `graph=True`인지 확인. 불일치시 종료 코드 1

> 5~7번은 GPU 없이 실행. OpenCL 추론과 모바일 수치 재현성은 별도 검증

## 8. MNNRuntime 빌드

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
* `--python`은 9번의 Python 추론 실행에 필요
* Runtime 출력: `MNNRuntime/build/MNN.dll`
* Python 바인딩: `MNNRuntime/python/_mnncengine*.pyd`
* 빌드 로그: `MNNRuntime/{configure,build,python-configure,python-build}.log`

## 9. C++ W8 모델로 이미지 추론

* 8번의 Windows OpenCL Runtime과 6번의 C++ 출력 모델을 사용

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
docker build -f Dockerfile.cpp.sd15 -t to-mnn:cpp-sd15 .
docker build -f Dockerfile.gen-template -t to-mnn:gen-template .
docker build -f Dockerfile.test.sd15 -t to-mnn:test-sd15 .
docker run --rm --network none to-mnn:test-sd15
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
