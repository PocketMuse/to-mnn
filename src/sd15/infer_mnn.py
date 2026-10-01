import argparse
import gc
import importlib
import importlib.util
import json
import os
import secrets
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np

BACKEND_IDS = {"CPU": 0, "OPENCL": 3}
SESSION_INFO_BACKENDS = 2
GPU_TUNING_NONE = 1 << 0
GPU_MEMORY_MODES = {"buffer": 1 << 6, "image": 1 << 7}
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_runtime(runtime_dir=None):
    """설치된 패키지 또는 직접 빌드한 Windows Runtime 바인딩을 읽는다."""
    if runtime_dir is None:
        return importlib.import_module("MNN")

    root = Path(runtime_dir).resolve()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    binding = root / "python" / f"_mnncengine{suffix}"
    dll_dir = root / "build"
    for path in (binding, dll_dir / "MNN.dll"):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run bash scripts/build-mnn.sh --python"
            )
    if os.name != "nt":
        raise RuntimeError("--runtime-dir requires the Windows native Runtime")

    loaded = sys.modules.get("MNN") or sys.modules.get("_mnncengine")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != binding:
            raise RuntimeError(
                "Another MNN binding is already loaded; use a new process"
            )
        return loaded

    dll_handle = os.add_dll_directory(str(dll_dir))
    try:
        spec = importlib.util.spec_from_file_location("_mnncengine", binding)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
    except Exception:
        dll_handle.close()
        raise

    # 공식 바인딩은 Session/Tensor 생성 시 MNN 모듈을 다시 찾는다.
    sys.modules["_mnncengine"] = runtime
    sys.modules["MNN"] = runtime
    runtime._dll_handle = dll_handle
    return runtime


def load_model(path, threads, backend="CPU", gpu_memory="buffer", runtime=None):
    """MNN 모델을 로딩하고 지정한 백엔드의 추론 세션을 생성한다.

    Args:
        path (Path): 로딩할 .mnn 파일 경로
        threads (int): CPU 추론에 사용할 스레드 수

    Returns:
        tuple: forward()에서 함께 사용할 (runtime, interpreter, session)
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    runtime = load_runtime() if runtime is None else runtime
    print(f"Loading {path.name} ({backend})", flush=True)
    interpreter = runtime.Interpreter(str(path))
    external = Path(str(path) + ".weight")
    if external.is_file():
        interpreter.setExternalFile(str(external))
    mode = (
        threads if backend == "CPU" else GPU_TUNING_NONE | GPU_MEMORY_MODES[gpu_memory]
    )
    session = interpreter.createSession(
        {"backend": backend, "numThread": mode, "precision": "high"}
    )
    actual = interpreter.getSessionInfo(session, SESSION_INFO_BACKENDS)
    if backend == "OPENCL" and actual != BACKEND_IDS[backend]:
        raise RuntimeError(
            f"Requested OPENCL, but {path.name} selected backend {actual}"
        )
    print(f"Session backend: {actual}", flush=True)
    return runtime, interpreter, session


def forward(model, inputs, output_name):
    """MNN 모델을 한 번 실행하고 지정한 출력을 NumPy 배열로 반환

    Args:
        model (tuple): load_model()이 반환한 (runtime, interpreter, session)
        inputs (dict[str, np.ndarray]): 모델 입력 이름과 NumPy 배열의 매핑
        output_name (str): 가져올 모델 출력 이름

    Returns:
        np.ndarray: NaN과 무한값이 없는 출력 배열

    Raises:
        ValueError: 입력 이름 또는 shape가 모델과 일치하지 않을 때
        TypeError: 입력 텐서의 dtype을 지원하지 않을 때
        RuntimeError: MNN 세션 실행이 실패할 때
        FloatingPointError: 출력에 NaN 또는 무한값이 포함될 때
    """
    MNN, interpreter, session = model
    tensors = interpreter.getSessionInputAll(session)

    if set(inputs) != set(tensors):
        raise ValueError(f"Input mismatch: {list(inputs)} vs {list(tensors)}")

    for name, value in inputs.items():
        target = tensors[name]
        if tuple(value.shape) != tuple(target.getShape()):
            raise ValueError(f"{name}: {value.shape} != {target.getShape()}")
        dtype = target.getDataType()
        if dtype == MNN.Halide_Type_Int:
            value = np.ascontiguousarray(value, dtype=np.int32)
        elif dtype == MNN.Halide_Type_Float:
            value = np.ascontiguousarray(value, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported input dtype for {name}: {dtype}")
        host = MNN.Tensor(value.shape, dtype, value, MNN.Tensor_DimensionType_Caffe)
        target.copyFrom(host)

    result = interpreter.runSession(session)

    if result != 0:
        raise RuntimeError(f"MNN runSession failed: {result}")

    output = interpreter.getSessionOutput(session, output_name)
    host = MNN.Tensor(
        output.getShape(),
        MNN.Halide_Type_Float,
        np.zeros(output.getShape(), dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    output.copyToHostTensor(host)
    array = host.getNumpyData().copy()

    if not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite MNN output: {output_name}")

    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt", default="a photo of a cat sitting on a wooden table, natural light"
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=BACKEND_IDS, default="CPU")
    parser.add_argument("--gpu-memory", choices=GPU_MEMORY_MODES, default="buffer")
    parser.add_argument(
        "--runtime-dir", type=Path, help="Native MNNRuntime directory (Windows)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" if os.name == "nt" else Path("/artifacts"),
    )
    parser.add_argument("--model-dir", type=Path, help="Defaults to artifacts-dir/mnn")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "img/mnn_sd15.png"
        if os.name == "nt"
        else Path("/img/mnn_sd15.png"),
    )
    args = parser.parse_args()

    if not 1 <= args.steps <= 999 or args.threads < 1:
        parser.error("steps must be 1..999 and threads must be positive")
    if args.seed is None:
        args.seed = secrets.randbits(32)

    print(f"Seed: {args.seed}", flush=True)

    if args.backend == "OPENCL" and os.name == "nt" and args.runtime_dir is None:
        args.runtime_dir = PROJECT_DIR / "MNNRuntime"
    runtime = load_runtime(args.runtime_dir)
    print(f"MNN binding: {runtime.__file__}", flush=True)
    model_options = {
        "threads": args.threads,
        "backend": args.backend,
        "gpu_memory": args.gpu_memory,
        "runtime": runtime,
    }

    # 시간 세팅, 모델 설정값 읽어오고 검사
    started = time.perf_counter()
    root = args.artifacts_dir
    args.model_dir = args.model_dir or root / "mnn"
    for name in ("text_encoder", "unet", "vae_decoder"):
        path = args.model_dir / f"{name}.mnn"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model: {path}; set --model-dir or --artifacts-dir"
            )
    cache = root / "huggingface/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision

    config = json.loads((snapshot / "scheduler/scheduler_config.json").read_text())
    vae_config = json.loads((snapshot / "vae/config.json").read_text())
    if config["beta_schedule"] != "scaled_linear" or config.get("clip_sample", False):
        raise ValueError(
            "This smoke runner expects the SD1.5 scaled-linear noise schedule"
        )
    if config.get("prediction_type", "epsilon") != "epsilon":
        raise ValueError("Only epsilon prediction is supported")

    # 두 프롬프트를 숫자로 변환
    from PIL import Image
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer", local_files_only=True
    )
    encoder = load_model(args.model_dir / "text_encoder.mnn", **model_options)
    embeddings = []
    for prompt in (args.negative_prompt, args.prompt):
        ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        ).input_ids
        embeddings.append(forward(encoder, {"input_ids": ids}, "last_hidden_state"))

    # 텍스트 인코더 참조 해제
    del encoder
    gc.collect()
    print(f"Text encoded: {embeddings[0].shape}", flush=True)

    # 노이즈 제거할 단계와 계산 계수를 준비
    train_steps = config["num_train_timesteps"]
    stride = train_steps // args.steps
    timesteps = np.arange(args.steps)[::-1] * stride + config.get("steps_offset", 0)

    betas = (
        np.linspace(
            config["beta_start"] ** 0.5,
            config["beta_end"] ** 0.5,
            train_steps,
            dtype=np.float32,
        )
        ** 2
    )
    alphas = np.cumprod(1 - betas)
    final_alpha = np.float32(1) if config.get("set_alpha_to_one", True) else alphas[0]

    # 초기 랜덤 latent를 만들고 UNet을 로딩
    latent = (
        np.random.default_rng(args.seed)
        .standard_normal((1, 4, 64, 64))
        .astype(np.float32)
    )
    unet = load_model(args.model_dir / "unet.mnn", **model_options)

    # 매 단계마다 UNet을 두번 실행하고 CFG를 계산
    for index, timestep in enumerate(timesteps):
        step_start = time.perf_counter()
        predictions = []
        for context in embeddings:
            # sample: 현재 노이즈가 섞인 latent
            # timestep: 현재 노이즈 단계
            # encoder_hidden_states: 프롬프트 임베딩
            predictions.append(
                forward(
                    unet,
                    {
                        "sample": latent,
                        "timestep": np.array([timestep], dtype=np.int32),
                        "encoder_hidden_states": context,
                    },
                    "out_sample",
                )
            )
        # CFG
        noise = predictions[0] + args.guidance * (predictions[1] - predictions[0])

        # DDIM 수식으로 다음 latent계산
        alpha = alphas[timestep]
        previous = timestep - stride
        alpha_previous = alphas[previous] if previous >= 0 else final_alpha

        # 현재 단계와 다음에 이동할 단계의 계수를 가져옴
        clean = (latent - np.sqrt(1 - alpha) * noise) / np.sqrt(alpha)

        # 현재 latent와 예측한 노이즈로부터 노이즈가 없는 latent를 추정
        latent = (
            np.sqrt(alpha_previous) * clean + np.sqrt(1 - alpha_previous) * noise
        ).astype(np.float32)

        if not np.isfinite(latent).all():
            raise FloatingPointError("Non-finite latent")
        print(
            f"Step {index + 1}/{args.steps}, t={timestep}, "
            f"{time.perf_counter() - step_start:.1f}s",
            flush=True,
        )

    del unet
    gc.collect()

    # 최종 latent를 이미지 텐서로 디코딩
    decoder = load_model(args.model_dir / "vae_decoder.mnn", **model_options)
    decoded = forward(
        decoder,
        {
            "latent_sample": latent
            / np.float32(vae_config.get("scaling_factor", 0.18215))
        },
        "sample",
    )

    if decoded.shape != (1, 3, 512, 512):
        raise ValueError(f"Unexpected decoded shape: {decoded.shape}")

    # 이미지 텐서를 픽셀로 바꾸고 저장
    pixels = np.clip(decoded[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    } | {
        "backend": f"MNN {args.backend}",
        "runtime_binding": runtime.__file__,
        "precision": "high",
        "scheduler": "DDIM eta=0",
        "config_revision": revision,
        "seconds": time.perf_counter() - started,
        "decoded_min": float(decoded.min()),
        "decoded_max": float(decoded.max()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {args.output} ({image.size}), {metadata['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
