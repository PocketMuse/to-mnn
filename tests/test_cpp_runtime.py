"""Windows C++ Runtime CLI의 도움말과 입력 검증을 검사한다."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("SD15_RUNTIME"), "C++ Runtime not selected")
class RuntimeCliTest(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            [os.environ["SD15_RUNTIME"], "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-dir", result.stdout)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--unet-precision", result.stdout)
        self.assertIn("--tuning", result.stdout)
        self.assertIn("--unet-low-memory", result.stdout)
        for model in ("clip", "unet", "vae"):
            for option in ("precision", "tuning", "low-memory"):
                self.assertIn(f"--{model}-{option}", result.stdout)
        self.assertNotIn("--input", result.stdout)

    def test_invalid_generation_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for flag, value, message in [
                ("--seed", "4294967296", "--seed"),
                ("--seed", "42junk", "--seed"),
                ("--seed", "-9223372036854775809", "--seed"),
                ("--steps", "0", "--steps"),
                ("--steps", "1000", "--steps"),
                ("--guidance", "nan", "finite"),
                ("--guidance", "inf", "finite"),
                ("--guidance", "7.5junk", "--guidance"),
                ("--input", "old.json", "--input"),
                ("--unet-precision", "int8", "--unet-precision"),
                ("--tuning", "invalid", "--tuning"),
            ]:
                with self.subTest(flag=flag, value=value):
                    result = subprocess.run(
                        [
                            os.environ["SD15_RUNTIME"],
                            "--model-dir",
                            root,
                            "--prompt",
                            "a cat",
                            "--output",
                            str(Path(root) / "image.png"),
                            flag,
                            value,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
            self.assertFalse((Path(root) / "image.png").exists())

    def test_valid_seeds_pass_parsing(self):
        for seed in [None, "-1", "-42", "-9223372036854775808", "0", "4294967295"]:
            with self.subTest(seed=seed):
                args = [os.environ["SD15_RUNTIME"]]
                if seed is not None:
                    args += ["--seed", seed]
                result = subprocess.run(args, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--model-dir", result.stderr)

    def test_runtime_options_pass_parsing(self):
        with tempfile.TemporaryDirectory() as root:
            for precision in ("high", "normal", "low"):
                for tuning in ("none", "fast", "normal", "wide"):
                    with self.subTest(precision=precision, tuning=tuning):
                        result = subprocess.run(
                            [
                                os.environ["SD15_RUNTIME"],
                                "--model-dir",
                                root,
                                "--prompt",
                                "",
                                "--output",
                                str(Path(root) / "out.png"),
                                "--unet-precision",
                                precision,
                                "--tuning",
                                tuning,
                            ],
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, 1)
                        self.assertIn("Missing model:", result.stderr)

    def test_model_options_are_independent_and_override_shared_tuning(self):
        with tempfile.TemporaryDirectory() as root:
            common = [
                os.environ["SD15_RUNTIME"],
                "--model-dir",
                root,
                "--prompt",
                "",
                "--output",
                str(Path(root) / "out.png"),
            ]
            overrides = [
                "--clip-precision",
                "normal",
                "--vae-precision",
                "low",
                "--unet-tuning",
                "none",
                "--vae-tuning",
                "wide",
                "--clip-low-memory",
                "--vae-low-memory",
            ]
            for flags in (
                ["--tuning", "fast"] + overrides,
                overrides + ["--tuning", "fast"],
            ):
                result = subprocess.run(common + flags, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    "clip requested: precision=normal, tuning=fast, low-memory=1",
                    result.stdout,
                )
                self.assertIn(
                    "unet requested: precision=high, tuning=none, low-memory=0",
                    result.stdout,
                )
                self.assertIn(
                    "vae requested: precision=low, tuning=wide, low-memory=1",
                    result.stdout,
                )
                self.assertNotIn("not expected", result.stderr)
            for model in ("clip", "unet", "vae"):
                for option in ("precision", "tuning"):
                    flag = f"--{model}-{option}"
                    result = subprocess.run(
                        common + [flag, "invalid"], capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(flag, result.stderr)
                    self.assertNotIn("requested:", result.stdout)

    def test_empty_and_unicode_prompts_pass_parsing(self):
        with tempfile.TemporaryDirectory() as root:
            for prompt in ("", "한글 고양이 🐈", "--a literal prompt"):
                with self.subTest(prompt=prompt):
                    result = subprocess.run(
                        [
                            os.environ["SD15_RUNTIME"],
                            "--model-dir",
                            root,
                            "--prompt",
                            prompt,
                            "--negative-prompt",
                            "",
                            "--output",
                            str(Path(root) / "결과.png"),
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("Missing model:", result.stderr)
            self.assertFalse((Path(root) / "결과.png").exists())

    def test_existing_unicode_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "결과.png"
            output.write_bytes(b"keep")
            result = subprocess.run(
                [
                    os.environ["SD15_RUNTIME"],
                    "--model-dir",
                    root,
                    "--prompt",
                    "",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Output already exists:", result.stderr)
            self.assertIn("결과.png", result.stderr)
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
