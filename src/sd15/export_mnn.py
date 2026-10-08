import argparse
import os
import shutil
import subprocess
from pathlib import Path

ONNX_DIR = Path("/artifacts/onnx")
MNN_DIR = Path("/artifacts/mnn")
COMPONENTS = ("text_encoder", "unet", "vae_decoder")
FLOAT_COMPONENTS = ("text_encoder", "vae_decoder")
HQQ_B128 = "hqq-b128"
CONVERT_OPTIONS = (
    "--transformerFuse",
    "--optimizePrefer",
    "2",
    "--optimizeLevel",
    "1",
)
HQQ_OPTIONS = (
    "--weightQuantBits",
    "8",
    "--weightQuantAsymmetric=1",
    "--weightQuantBlock",
    "128",
    "--hqq",
)
HQQ_LOG_MARKER = "Use HQQ to quant weight"


def export_component(name, onnx_dir=ONNX_DIR, output_dir=MNN_DIR, *, hqq=False):
    """모델 하나를 변환하고 로그와 출력 파일로 성공 여부를 확인

    Args:
        name (str): ONNX 하위 폴더와 MNN 출력 파일에 사용할 모델 이름

    Returns:
        Path: 생성된 .mnn 파일 경로
    """
    source = onnx_dir / name / "model.onnx"
    output = output_dir / f"{name}.mnn"
    log_path = output_dir / f"{name}.convert.log"

    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # MNN 로그 모듈이 시도하는 부가 SDK의 pip 다운로드를 차단
    env = os.environ.copy()
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    print(f"Converting {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        # Run MNN Convert
        result = subprocess.run(
            [
                "mnnconvert",
                "-f",
                "ONNX",
                "--modelFile",
                str(source),
                "--MNNModel",
                str(output),
                *CONVERT_OPTIONS,
                *(HQQ_OPTIONS if hqq else ("--fp16",)),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    print(log_text, end="", flush=True)

    if result.returncode != 0 or "Converted Success!" not in log_text:
        raise RuntimeError(f"MNN conversion failed: {name}; see {log_path}")
    if hqq and HQQ_LOG_MARKER not in log_text:
        raise RuntimeError(f"HQQ was not enabled: {name}; see {log_path}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MNN output missing or empty: {output}")

    return output


def export_w8(onnx_dir, reference_dir, output_dir):
    """FP16 CLIP/VAE를 복사하고 HQQ B128 UNet 기준 모델을 생성한다."""
    if output_dir.resolve() == reference_dir.resolve():
        raise ValueError("W8 output must differ from the FP16 reference directory")

    required = [onnx_dir / "unet" / "model.onnx"]
    for name in FLOAT_COMPONENTS:
        model = reference_dir / f"{name}.mnn"
        required.append(model)
        external = model.with_suffix(".mnn.weight")
        if external.exists():
            required.append(external)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"W8 input missing or empty: {path}")

    output = export_component("unet", onnx_dir, output_dir, hqq=True)
    print(f"Saved {output}", flush=True)
    for name in FLOAT_COMPONENTS:
        model = reference_dir / f"{name}.mnn"
        shutil.copyfile(model, output_dir / model.name)
        external = model.with_suffix(".mnn.weight")
        destination = output_dir / external.name
        if external.is_file():
            shutil.copyfile(external, destination)
        else:
            # 이전 실행의 외부 가중치가 새 모델에 연결되지 않게 한다.
            destination.unlink(missing_ok=True)
        print(f"Copied {output_dir / model.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert SD1.5 ONNX models to MNN")
    parser.add_argument("--onnx-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=MNN_DIR)
    parser.add_argument("--unet-quantization", choices=(HQQ_B128,))
    parser.add_argument("--reference-mnn-dir", type=Path, default=MNN_DIR)
    args = parser.parse_args()

    if args.unet_quantization == HQQ_B128:
        export_w8(args.onnx_dir, args.reference_mnn_dir, args.output_dir)
        return

    for name in COMPONENTS:
        source = args.onnx_dir / name / "model.onnx"
        if not source.is_file():
            raise FileNotFoundError(f"ONNX model not found: {source}")

    for name in COMPONENTS:
        output = export_component(name, args.onnx_dir, args.output_dir)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
