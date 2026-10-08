import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from safetensors.numpy import save_file

from src.sd15.gen_template import classify, generate
from src.sd15.mnn_tensors import MnnModel
from src.sd15.template_values import COMPONENTS
from tests.test_mnn_compare import write_model, write_quant_model


class TemplateGenerationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.onnx = self.root / "onnx"
        self.mnn = self.root / "mnn"
        self.mnn.mkdir()
        self.checkpoint = self.root / "reference.safetensors"
        self.output = self.root / "template"
        self.weights = {}
        for i, component in enumerate(COMPONENTS):
            values = np.array([i * 2, i * 3], dtype="<f4")
            self.weights[component + ".weight"] = values
            (self.onnx / component).mkdir(parents=True)
            graph = onnx.helper.make_graph(
                [],
                component,
                [],
                [
                    onnx.helper.make_tensor_value_info(
                        "weight", onnx.TensorProto.FLOAT, [2]
                    )
                ],
                [onnx.numpy_helper.from_array(values, name="weight")],
            )
            onnx.save(
                onnx.helper.make_model(graph), self.onnx / component / "model.onnx"
            )
            write_model(self.mnn / (component + ".mnn"), values)
        save_file(self.weights, str(self.checkpoint))

    def test_missing_model_fails_before_indexing_weights(self):
        missing = self.mnn / "text_encoder.mnn"
        missing.unlink()
        with patch("src.sd15.gen_template.index_source") as index:
            with self.assertRaisesRegex(
                FileNotFoundError, "Template input not found"
            ) as error:
                generate(self.checkpoint, self.onnx, self.mnn, self.output)
        self.assertIn(str(missing), str(error.exception))
        index.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_zero_valued_learned_tensor_remains_a_parameter(self):
        generate(self.checkpoint, self.onnx, self.mnn, self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())
        tensors = [
            s for f in manifest["files"] for s in f["segments"] if s["kind"] == "tensor"
        ]
        self.assertEqual({s["key"] for s in tensors}, set(self.weights))
        self.assertEqual(len(tensors), 3)

    def test_fused_attention_is_preserved_in_template(self):
        path = self.mnn / "text_encoder.mnn"
        write_model(path, [0, 0], attention_scale=0.125)
        generate(self.checkpoint, self.onnx, self.mnn, self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())
        graph = (self.output / "graph.bin").read_bytes()
        output = next(f for f in manifest["files"] if f["name"] == path.name)
        restored = bytearray()
        for segment in output["segments"]:
            if segment["kind"] == "literal":
                start = segment["template_offset"]
                restored.extend(graph[start : start + segment["size"]])
            else:
                restored.extend(bytes(segment["size"]))
        self.assertEqual(restored, path.read_bytes())

    def test_ambiguous_mapping_fails_and_removes_only_new_output(self):
        self.weights["duplicate"] = self.weights["unet.weight"].copy()
        save_file(self.weights, str(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "expected one source"):
            generate(self.checkpoint, self.onnx, self.mnn, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.checkpoint.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            generate(self.checkpoint, self.onnx, self.mnn, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_quantized_payload_and_alpha_share_source_mapping(self):
        path = self.mnn / "unet.mnn"
        write_quant_model(path)
        provenance = {("conv", "Weight"): {"key": "weight", "shape": [1, 2, 1, 1]}}
        with MnnModel(path) as model:
            regions = classify(
                model, {"weight": {}}, {}, {"weight": [1, 2]}, provenance
            )
        outputs = regions[path.name]
        self.assertEqual({s["field"] for s in outputs}, {"Weight", "Alpha"})
        self.assertTrue(all(s["kind"] == "quantized" for s in outputs))
        self.assertTrue(all(s["key"] == "weight" for s in outputs))
        self.assertTrue(all(s["quantization"]["algorithm"] == "hqq" for s in outputs))

    def test_unmapped_mnn_tensor_fails_without_leaking_mmap_views(self):
        write_model(self.mnn / "text_encoder.mnn", [9, 10])
        with self.assertRaisesRegex(ValueError, "Missing MNN parameters"):
            generate(self.checkpoint, self.onnx, self.mnn, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
