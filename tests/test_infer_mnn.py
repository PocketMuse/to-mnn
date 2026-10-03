import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.sd15.infer_mnn import forward, load_model, load_runtime


class BackendSelectionTest(unittest.TestCase):
    def test_opencl_cpu_fallback_is_rejected(self):
        runtime = Mock()
        runtime.Interpreter.return_value.getSessionInfo.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mnn"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "OPENCL.*0"):
                load_model(path, 4, backend="OPENCL", runtime=runtime)

    def test_missing_native_runtime_does_not_load_pip_package(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_runtime(Path(directory))


@unittest.skipUnless(os.environ.get("SD15_MNN_RUNTIME"), "Native Runtime not selected")
class NativeInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime(Path(os.environ["SD15_MNN_RUNTIME"]))

    def test_matmul_cpu_and_opencl(self):
        expression = self.runtime._expr
        a = expression.placeholder([2, 3], expression.NCHW, expression.float)
        b = expression.placeholder([3, 2], expression.NCHW, expression.float)
        a.name, b.name = "a", "b"
        output = expression.matmul(a, b, False, False)
        output.name = "product"
        inputs = {
            "a": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
            "b": np.arange(1, 7, dtype=np.float32).reshape(3, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("CPU", "OPENCL"):
                with self.subTest(backend=backend):
                    path = Path(directory) / f"{backend}.mnn"
                    expression.save([output], str(path))
                    model = load_model(
                        path,
                        4,
                        backend=backend,
                        runtime=self.runtime,
                    )
                    result = forward(model, inputs, "product")
                    self.assertEqual(result.shape, (2, 2))
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_allclose(
                        result, inputs["a"] @ inputs["b"], rtol=1e-5
                    )
                    with self.assertRaisesRegex(ValueError, "Input mismatch"):
                        forward(model, {"wrong": inputs["a"]}, "product")
                    del model


if __name__ == "__main__":
    unittest.main()
