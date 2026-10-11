"""Windows C++ Runtime CLI의 도움말과 입력 검증을 검사한다."""

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


if __name__ == "__main__":
    unittest.main()
