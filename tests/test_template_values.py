import unittest

import numpy as np

from src.sd15.template_values import encode_tensor


class TensorEncodingTest(unittest.TestCase):
    def test_mnn_half_truncates_and_clamps(self):
        values = np.array([1.0008, -1.0008, 1e10, -1e10, 2**-24], dtype="<f4")
        actual = np.frombuffer(encode_tensor(values, "F16", False), dtype="<u2")
        np.testing.assert_array_equal(actual, [0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 1])

    def test_only_source_zero_is_normalized(self):
        values = np.array([-0.0, -(2**-25)], dtype="<f4")
        actual = np.frombuffer(encode_tensor(values, "F16", True), dtype="<u2")
        np.testing.assert_array_equal(actual, [0, 0x8000])

    def test_float32_preserves_bits(self):
        values = np.array([-0.0, 1.0008], dtype="<f4")
        self.assertEqual(encode_tensor(values, "F32", False), values.tobytes())

    def test_scalar_tensor(self):
        self.assertEqual(
            encode_tensor(np.array(1.0, dtype="<f4"), "F16", False), b"\x00\x3c"
        )

    def test_nonfinite_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_tensor(np.array([np.nan], dtype="<f4"), "F16", False)


if __name__ == "__main__":
    unittest.main()
