"""C++ 공개 API의 상태·취소·오류 계약을 검사한다."""

import os
import subprocess
import tempfile
import unittest


class ConverterApiTest(unittest.TestCase):
    def test_native_api_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    os.environ.get("SD15_API_TEST", "/usr/local/bin/sd15-api-test"),
                    directory,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
