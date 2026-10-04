from pathlib import Path
from types import MethodType

import onnx
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention, AttnProcessor

from .prepare_runtime_cache import prepare_runtime_cache

CHECKPOINT = Path("/models/v1-5-pruned-emaonly.safetensors")
CONFIG_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CACHE_DIR = Path("/artifacts/huggingface")
ONNX_DIR = Path("/artifacts/onnx")


def unmasked_attention_scores(self, query, key, attention_mask=None):
    """export 시 beta=0의 거대한 0 상수와 Add 생성을 피한다."""
    if attention_mask is not None:
        return Attention.get_attention_scores(self, query, key, attention_mask)

    dtype = query.dtype
    if self.upcast_attention:
        query, key = query.float(), key.float()
    scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if self.upcast_softmax:
        scores = scores.float()
    return scores.softmax(dim=-1).to(dtype)


def prepare_unet_attention(unet):
    """UNet 인스턴스의 export용 Attention만 교체한다."""
    for module in unet.modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor())
            module.get_attention_scores = MethodType(unmasked_attention_scores, module)


class TextEncoderExport(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(
            input_ids=input_ids,
            return_dict=False,
        )[0]


class UNetExport(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VAEDecoderExport(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample, return_dict=False)[0]


def load_pipeline() -> StableDiffusionPipeline:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    pipeline = StableDiffusionPipeline.from_single_file(
        str(CHECKPOINT),
        config=CONFIG_REPO,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipeline.to("cpu")

    for name in ("text_encoder", "unet", "vae"):
        model = getattr(pipeline, name)
        model.eval()
        model.requires_grad_(False)

        parameter_count = sum(p.numel() for p in model.parameters())
        parameter = next(model.parameters())

        print(
            f"{name}: "
            f"{parameter_count:,} parameters, "
            f"dtype={parameter.dtype}, "
            f"device={parameter.device}"
        )

    return pipeline


def export_pipeline(pipeline):
    @torch.no_grad()
    def export_component(model, inputs, name, input_names, output_names):
        output_path = ONNX_DIR / name / "model.onnx"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamo=False,
        )

        onnx.checker.check_model(str(output_path))
        print(f"ONNX check passed: {name}", flush=True)

    torch.manual_seed(0)

    pipeline.text_encoder.set_attn_implementation("eager")
    prepare_unet_attention(pipeline.unet)
    pipeline.vae.set_attn_processor(AttnProcessor())

    batch_size = 1
    latent_size = 64
    sequence_length = pipeline.tokenizer.model_max_length
    hidden_size = pipeline.text_encoder.config.hidden_size

    tokens = pipeline.tokenizer(
        ["a photo of a cat"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(dtype=torch.int32)

    export_component(
        TextEncoderExport(pipeline.text_encoder),
        (tokens,),
        "text_encoder",
        ["input_ids"],
        ["last_hidden_state"],
    )

    sample = torch.randn(batch_size, 4, latent_size, latent_size)
    timestep = torch.tensor([500], dtype=torch.int64)
    encoder_hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
    )

    export_component(
        UNetExport(pipeline.unet),
        (sample, timestep, encoder_hidden_states),
        "unet",
        ["sample", "timestep", "encoder_hidden_states"],
        ["out_sample"],
    )

    latent_sample = torch.randn(batch_size, 4, latent_size, latent_size)

    export_component(
        VAEDecoderExport(pipeline.vae),
        (latent_sample,),
        "vae_decoder",
        ["latent_sample"],
        ["sample"],
    )


def main() -> None:
    pipeline = load_pipeline()
    prepare_runtime_cache()

    print("UNet input channels:", pipeline.unet.config.in_channels)
    print("UNet sample size:", pipeline.unet.config.sample_size)
    print("Text max length:", pipeline.tokenizer.model_max_length)
    print("VAE scaling factor:", pipeline.vae.config.scaling_factor)

    export_pipeline(pipeline)


if __name__ == "__main__":
    main()
