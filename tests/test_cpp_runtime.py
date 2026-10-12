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
        self.assertNotIn("--input", result.stdout)

    def test_invalid_generation_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for flag, value, message in [
                ("--seed", "4294967296", "seed must"),
                ("--seed", "42junk", "seed must"),
                ("--seed", "-9223372036854775809", "seed must"),
                ("--steps", "0", "steps must"),
                ("--steps", "1000", "steps must"),
                ("--guidance", "nan", "finite"),
                ("--guidance", "7.5junk", "finite"),
                ("--input", "old.json", "Unknown option"),
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
                self.assertIn("Require --model-dir", result.stderr)


if __name__ == "__main__":
    unittest.main()
