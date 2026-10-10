"""작은 safetensors로 스트리밍·dtype·실패 처리를 검증한다."""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = self.root / "template"
        self.template.mkdir()
        (self.template / "graph.bin").write_bytes(b"MNN!END")
        self.checkpoint = self.root / "model.safetensors"
        self.output = self.root / "out"
        self.values = [1.0008, -1.0008, -0.0, 2**-24, -(2**-25), 1e10]
        self.write_source("F32", struct.pack("<6f", *self.values))
        self.manifest = {
            "format": "sd15-mnn-template",
            "version": 1,
            "mnn_version": "3.6.1",
            "template": "graph.bin",
            "template_size": 7,
            "files": [
                {
                    "name": "test.mnn",
                    "size": 38,
                    "segments": [
                        {
                            "kind": "literal",
                            "offset": 0,
                            "size": 4,
                            "template_offset": 0,
                        },
                        {
                            "kind": "tensor",
                            "offset": 4,
                            "size": 12,
                            "key": "weight",
                            "source_shape": [2, 3],
                            "shape": [6],
                            "dtype": "F16",
                            "positive_zero": True,
                        },
                        {"kind": "zero", "offset": 16, "size": 19},
                        {
                            "kind": "literal",
                            "offset": 35,
                            "size": 3,
                            "template_offset": 4,
                        },
                    ],
                }
            ],
        }

    def write_source(self, dtype, data, shape=None):
        header = json.dumps(
            {
                "weight": {
                    "dtype": dtype,
                    "shape": shape or [2, 3],
                    "data_offsets": [0, len(data)],
                }
            }
        ).encode()
        self.checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + data)

    def run_converter(self, succeeds=True, iterations=None):
        (self.template / "manifest.json").write_text(json.dumps(self.manifest))
        result = subprocess.run(
            [
                os.environ.get("SD15_CONVERTER", "/usr/local/bin/sd15-convert"),
                "--checkpoint",
                str(self.checkpoint),
                "--template-dir",
                str(self.template),
                "--output",
                str(self.output),
                "--chunk-bytes",
                "4",
            ]
            + ([] if iterations is None else ["--hqq-iterations", str(iterations)]),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stderr)
        if not succeeds:
            self.assertFalse(self.output.exists())
            self.assertFalse(Path(str(self.output) + ".partial").exists())
        return result

    def test_streams_tensor_and_zero_runs_across_chunks(self):
        self.run_converter()
        half = struct.pack("<6H", 0x3C00, 0xBC00, 0, 1, 0x8000, 0x7BFF)
        self.assertEqual(
            (self.output / "test.mnn").read_bytes(), b"MNN!" + half + bytes(19) + b"END"
        )

    def test_reads_new_weights_instead_of_reference_weights(self):
        self.write_source("F32", struct.pack("<6f", 1, 2, 3, 4, 5, 6))
        self.run_converter()
        self.assertEqual(
            (self.output / "test.mnn").read_bytes()[4:16],
            struct.pack("<6e", 1, 2, 3, 4, 5, 6),
        )

    def test_accepts_float16_source(self):
        self.write_source("F16", struct.pack("<6e", 1, 2, 3, 4, 5, 6))
        self.run_converter()
        self.assertEqual(
            (self.output / "test.mnn").read_bytes()[4:16],
            struct.pack("<6e", 1, 2, 3, 4, 5, 6),
        )

    def configure_hqq(self, values, groups=1):
        self.write_source(
            "F32", struct.pack(f"<{len(values)}f", *values), [len(values)]
        )
        area = len(values) // groups
        recipe = {
            "algorithm": "hqq",
            "bits": 8,
            "group_elements": area,
            "group_count": groups,
            "iterations": 20,
            "lp_norm": 0.7,
            "beta": 10.0,
        }
        size = groups * 8 + len(values)
        segments = [
            {
                "kind": "quantized",
                "field": "Alpha",
                "dtype": "F32",
                "shape": [groups, 2],
                "offset": 0,
                "size": groups * 8,
            },
            {
                "kind": "quantized",
                "field": "Weight",
                "dtype": "U8",
                "shape": [len(values)],
                "offset": groups * 8,
                "size": len(values),
            },
        ]
        for segment in segments:
            segment.update(
                key="weight",
                source_shape=[len(values)],
                positive_zero=False,
                quantization=recipe,
            )
        self.manifest.update(version=2)
        self.manifest["files"] = [
            {"name": "test.mnn", "size": size, "segments": segments}
        ]

    def test_hqq_recreates_payload_and_scales(self):
        self.configure_hqq([-1, 1])
        self.run_converter()
        data = (self.output / "test.mnn").read_bytes()
        self.assertEqual(data[8:], b"\x00\xff")
        minimum, scale = struct.unpack("<2f", data[:8])
        self.assertAlmostEqual(minimum, -1, places=6)
        self.assertAlmostEqual(scale, 2 / 255, places=8)

    def test_hqq_zero_skips_refinement_and_default_matches_twenty(self):
        self.configure_hqq([0, 0.49, 1.49, 255])
        outputs = {}
        for count in (None, 20, 15, 10, 5, 0):
            self.output = self.root / f"out-{count}"
            result = self.run_converter(iterations=count)
            self.assertIn(
                f"hqq_iterations={20 if count is None else count}", result.stdout
            )
            self.assertIn("hqq_compute_ms=", result.stdout)
            outputs[count] = (self.output / "test.mnn").read_bytes()
        self.assertEqual(outputs[None], outputs[20])
        minimum, scale = struct.unpack("<2f", outputs[0][:8])
        self.assertEqual(minimum, 0)
        self.assertEqual(scale, 1)
        self.assertEqual(outputs[0][8:], bytes([0, 0, 1, 255]))
        self.assertNotEqual(outputs[0], outputs[20])

    def test_hqq_rejects_invalid_iteration_options(self):
        self.configure_hqq([-1, 1])
        for count in (-1, 21, "1.5", "oops"):
            with self.subTest(iterations=count):
                self.run_converter(False, iterations=count)

    def test_hqq_reads_replacement_weights(self):
        self.configure_hqq([-2, 2, -1, 1], groups=2)
        self.run_converter()
        data = (self.output / "test.mnn").read_bytes()
        self.assertEqual(data[16:], b"\x00\xff\x00\xff")
        self.assertAlmostEqual(struct.unpack("<f", data[:4])[0], -2, places=6)

    def test_hqq_rejects_missing_alpha(self):
        self.configure_hqq([-1, 1])
        weight = self.manifest["files"][0]["segments"][1]
        weight["offset"] = 0
        self.manifest["files"][0].update(size=2, segments=[weight])
        self.run_converter(False)

    def test_hqq_rejects_nonfinite_source(self):
        self.configure_hqq([-1, float("nan")])
        self.run_converter(False)

    def test_hqq_rejects_range_overflow(self):
        self.configure_hqq([-3e38, 3e38])
        self.assertIn("overflow", self.run_converter(False).stderr)

    def test_hqq_rejects_normalization_overflow(self):
        self.configure_hqq([3e38, 3e38])
        self.assertIn("overflow", self.run_converter(False).stderr)

    def test_hqq_accepts_float16_source(self):
        self.configure_hqq([-1, 1])
        self.write_source("F16", struct.pack("<2e", -1, 1), [2])
        self.run_converter()
        self.assertEqual((self.output / "test.mnn").read_bytes()[8:], b"\x00\xff")

    def test_hqq_rejects_inconsistent_group_recipe(self):
        self.configure_hqq([-1, 1])
        self.manifest["files"][0]["segments"][1]["quantization"] = {
            **self.manifest["files"][0]["segments"][1]["quantization"],
            "group_elements": 1,
        }
        self.run_converter(False)

    def test_all_finite_half_values_expand_exactly(self):
        bits = [i for i in range(65536) if i & 0x7C00 != 0x7C00]
        data = struct.pack(f"<{len(bits)}H", *bits)
        self.write_source("F16", data, [len(bits)])
        size = len(bits) * 4
        segment = self.manifest["files"][0]["segments"][1]
        segment.update(
            dtype="F32",
            size=size,
            source_shape=[len(bits)],
            shape=[len(bits)],
            positive_zero=False,
            offset=0,
        )
        self.manifest["files"][0].update(size=size, segments=[segment])
        self.run_converter()
        expected = struct.pack(f"<{len(bits)}f", *struct.unpack(f"<{len(bits)}e", data))
        self.assertEqual((self.output / "test.mnn").read_bytes(), expected)

    def test_float32_storage_preserves_signed_zero(self):
        data = struct.pack("<6f", *self.values)
        segment = self.manifest["files"][0]["segments"][1]
        segment.update(dtype="F32", size=len(data), positive_zero=False, offset=0)
        self.manifest["files"][0].update(size=len(data), segments=[segment])
        self.run_converter()
        self.assertEqual((self.output / "test.mnn").read_bytes(), data)

    def test_rejects_missing_key(self):
        self.manifest["files"][0]["segments"][1]["key"] = "missing"
        self.assertIn("missing", self.run_converter(False).stderr)

    def test_rejects_wrong_shape(self):
        self.write_source("F32", struct.pack("<6f", *self.values), [3, 2])
        self.assertIn("shape", self.run_converter(False).stderr)

    def test_rejects_truncated_data(self):
        self.checkpoint.write_bytes(self.checkpoint.read_bytes()[:-1])
        self.run_converter(False)

    def test_rejects_tensor_length_mismatch(self):
        self.manifest["files"][0]["segments"][1]["size"] = 10
        self.run_converter(False)

    def test_rejects_output_path_traversal(self):
        self.manifest["files"][0]["name"] = "../escaped.mnn"
        self.run_converter(False)
        self.assertFalse((self.root / "escaped.mnn").exists())

    def test_rejects_nonfinite_weight_and_cleans_partial_files(self):
        self.write_source("F32", struct.pack("<6f", 1, 2, 3, 4, 5, float("nan")))
        self.run_converter(False)

    def test_rejects_oversized_header_before_allocating(self):
        self.checkpoint.write_bytes(struct.pack("<Q", 2**40))
        self.run_converter(False)

    def test_rejects_duplicate_json_keys(self):
        header = b'{"weight": {}, "weight": {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}'
        self.checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
        self.run_converter(False)


if __name__ == "__main__":
    unittest.main()
