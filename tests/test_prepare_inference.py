import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.sd15.prepare_inference import prepare


class PrepareInferenceTest(unittest.TestCase):
    def test_missing_vae_scale_uses_sd15_default(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts = Path(root)
            cache = (
                artifacts
                / "huggingface/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
            )
            (cache / "refs").mkdir(parents=True)
            (cache / "refs/main").write_text("test")
            snapshot = cache / "snapshots/test"
            (snapshot / "scheduler").mkdir(parents=True)
            (snapshot / "vae").mkdir()
            config = {
                "beta_schedule": "scaled_linear",
                "prediction_type": "epsilon",
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "steps_offset": 1,
                "set_alpha_to_one": False,
                "beta_start": 0.00085,
                "beta_end": 0.012,
            }
            (snapshot / "scheduler/scheduler_config.json").write_text(
                json.dumps(config)
            )
            (snapshot / "vae/config.json").write_text("{}")
            with patch(
                "src.sd15.prepare_inference.CLIPTokenizer.from_pretrained"
            ) as load:
                load.return_value.return_value = SimpleNamespace(
                    input_ids=np.zeros((2, 77), dtype=np.int32)
                )
                result = prepare(artifacts, "cat", "", 42, 20, 7.5)
            self.assertEqual(result["vae_scaling_factor"], 0.18215)
            self.assertEqual(len(result["initial_latent"]), 16384)
            self.assertAlmostEqual(result["initial_latent"][0], 0.3047170798, places=6)
            self.assertEqual(len(result["tokens"]), 77)
            config["prediction_type"] = "v_prediction"
            (snapshot / "scheduler/scheduler_config.json").write_text(
                json.dumps(config)
            )
            with self.assertRaisesRegex(ValueError, "prediction_type"):
                prepare(artifacts, "cat", "", 42, 20, 7.5)

    def test_invalid_steps_and_guidance_are_rejected_before_reading(self):
        for steps, guidance in [(0, 7.5), (1000, 7.5), (20, float("nan"))]:
            with (
                self.subTest(steps=steps, guidance=guidance),
                self.assertRaises(ValueError),
            ):
                prepare(Path("missing"), "cat", "", 42, steps, guidance)


if __name__ == "__main__":
    unittest.main()
