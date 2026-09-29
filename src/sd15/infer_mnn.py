import argparse
import gc
import json
import secrets
import time
from pathlib import Path

import MNN
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer


def load_model(path, threads):
    """MNN 모델을 로딩하고 CPU 추론 세션을 생성

    Args:
        path (Path): 로딩할 .mnn 파일 경로
        threads (int): CPU 추론에 사용할 스레드 수

    Returns:
        tuple: forward()에서 함께 사용할 (interpreter, session)
    """
    print(f"Loading {path.name}", flush=True)
    interpreter = MNN.Interpreter(str(path))
    external = Path(str(path) + ".weight")
    if external.is_file():
        interpreter.setExternalFile(str(external))
    session = interpreter.createSession(
        {"backend": "CPU", "numThread": threads, "precision": "high"}
    )
    return interpreter, session


def forward(model, inputs, output_name):
    """MNN 모델을 한 번 실행하고 지정한 출력을 NumPy 배열로 반환

    Args:
        model (tuple): load_model()이 반환한 (interpreter, session)
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
    interpreter, session = model
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
    parser.add_argument("--output", type=Path, default=Path("/img/mnn_sd15.png"))
    args = parser.parse_args()

    if not 1 <= args.steps <= 999 or args.threads < 1:
        parser.error("steps must be 1..999 and threads must be positive")
    if args.seed is None:
        args.seed = secrets.randbits(32)

    print(f"Seed: {args.seed}", flush=True)

    # 시간 세팅, 모델 설정값 읽어오고 검사
    started = time.perf_counter()
    root = Path("/artifacts")
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
    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer", local_files_only=True
    )
    encoder = load_model(root / "mnn/text_encoder.mnn", args.threads)
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
    unet = load_model(root / "mnn/unet.mnn", args.threads)

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
    decoder = load_model(root / "mnn/vae_decoder.mnn", args.threads)
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
    metadata = vars(args) | {
        "output": str(args.output),
        "backend": "MNN CPU",
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
