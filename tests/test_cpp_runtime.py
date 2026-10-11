"""Windows C++ Runtime CLI와 동일 입력의 Python 기준 이미지를 검증한다."""

import json
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

    def test_unknown_input_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            request = Path(root) / "input.json"
            request.write_text(json.dumps({"format": "wrong", "version": 1}))
            result = subprocess.run(
                [
                    os.environ["SD15_RUNTIME"],
                    "--model-dir",
                    root,
                    "--input",
                    str(request),
                    "--output",
                    str(Path(root) / "image.png"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported inference input", result.stderr)
            self.assertFalse((Path(root) / "image.png").exists())

    @unittest.skipUnless(
        all(
            os.environ.get(key)
            for key in (
                "SD15_RUNTIME_MODEL",
                "SD15_RUNTIME_INPUT",
                "SD15_RUNTIME_REFERENCE",
            )
        ),
        "Real model, input and reference PNG not selected",
    )
    def test_image_matches_python_reference(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "image.png"
            result = subprocess.run(
                [
                    os.environ["SD15_RUNTIME"],
                    "--model-dir",
                    os.environ["SD15_RUNTIME_MODEL"],
                    "--input",
                    os.environ["SD15_RUNTIME_INPUT"],
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (
                Image.open(output) as actual,
                Image.open(os.environ["SD15_RUNTIME_REFERENCE"]) as expected,
            ):
                self.assertEqual(actual.size, (512, 512))
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()
