"""C++ 추론 검증용 CLIP token IDs와 초기 latent를 준비한다."""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import CLIPTokenizer


def prepare(artifacts, prompt, negative_prompt, seed, steps, guidance):
    if not 1 <= steps <= 999 or not np.isfinite(guidance):
        raise ValueError("Expected steps in [1, 999] and finite guidance")
    cache = (
        artifacts / "huggingface/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    )
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision
    config = json.loads((snapshot / "scheduler/scheduler_config.json").read_text())
    expected = {
        "beta_schedule": "scaled_linear",
        "prediction_type": "epsilon",
        "clip_sample": False,
        "num_train_timesteps": 1000,
        "steps_offset": 1,
        "set_alpha_to_one": False,
        "beta_start": 0.00085,
        "beta_end": 0.012,
    }
    for key, value in expected.items():
        if config.get(key, "epsilon" if key == "prediction_type" else None) != value:
            raise ValueError(f"Unsupported scheduler setting: {key}")
    vae = json.loads((snapshot / "vae/config.json").read_text())
    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer", local_files_only=True
    )
    tokens = tokenizer(
        [negative_prompt, prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    ).input_ids
    latent = (
        np.random.default_rng(seed).standard_normal((1, 4, 64, 64)).astype(np.float32)
    )
    return {
        "format": "sd15-inference-input",
        "version": 1,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "config_revision": revision,
        "scheduler": expected,
        "vae_scaling_factor": vae.get("scaling_factor", 0.18215),
        "tokens": tokens[1].tolist(),
        "negative_tokens": tokens[0].tolist(),
        "initial_latent": latent.reshape(-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.artifacts_dir,
        args.prompt,
        args.negative_prompt,
        args.seed,
        args.steps,
        args.guidance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
