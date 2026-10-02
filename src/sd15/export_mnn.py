import argparse
import os
import subprocess
from pathlib import Path

ONNX_DIR = Path("/artifacts/onnx")
MNN_DIR = Path("/artifacts/mnn")


def export_component(
    name, onnx_dir=ONNX_DIR, output_dir=MNN_DIR, transformer_fuse=False
):
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
                "--fp16",
                *(["--transformerFuse"] if transformer_fuse else []),
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
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MNN output missing or empty: {output}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert SD1.5 ONNX models to MNN")
    parser.add_argument("--onnx-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=MNN_DIR)
    parser.add_argument("--transformer-fuse", action="store_true")
    args = parser.parse_args()

    for name in ("text_encoder", "unet", "vae_decoder"):
        source = args.onnx_dir / name / "model.onnx"
        if not source.is_file():
            raise FileNotFoundError(f"ONNX model not found: {source}")

    for name in ("text_encoder", "unet", "vae_decoder"):
        output = export_component(
            name, args.onnx_dir, args.output_dir, args.transformer_fuse
        )
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
