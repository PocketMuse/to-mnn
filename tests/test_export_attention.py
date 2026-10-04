import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor

from src.sd15.export_onnx import prepare_unet_attention


class ExportAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.reference = Attention(query_dim=16, heads=2, dim_head=8).eval()
        self.reference.set_processor(AttnProcessor())
        self.candidate = copy.deepcopy(self.reference)
        prepare_unet_attention(self.candidate)

    def test_attention_outputs_match_with_and_without_mask(self):
        query = torch.randn(2, 16, 8)
        key = torch.randn(2, 7, 8)
        mask = torch.zeros(2, 16, 7)
        mask[:, :, -1] = -10000
        for attention_mask in (None, mask):
            for upcast in (False, True):
                with self.subTest(mask=attention_mask is not None, upcast=upcast):
                    self.reference.upcast_attention = upcast
                    self.reference.upcast_softmax = upcast
                    self.candidate.upcast_attention = upcast
                    self.candidate.upcast_softmax = upcast
                    expected = self.reference.get_attention_scores(
                        query, key, attention_mask
                    )
                    actual = self.candidate.get_attention_scores(
                        query, key, attention_mask
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_export_has_no_attention_sized_constants(self):
        hidden = torch.randn(1, 32, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.onnx"
            torch.onnx.export(
                self.candidate,
                (hidden,),
                str(path),
                input_names=["hidden"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
                do_constant_folding=True,
            )
            model = onnx.load(path)
            onnx.checker.check_model(model)
            for node in model.graph.node:
                for attribute in node.attribute:
                    if attribute.HasField("t"):
                        self.assertLess(np.prod(attribute.t.dims), 32, node.name)
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, {"hidden": hidden.numpy()})[0]
            with torch.no_grad():
                expected = self.reference(hidden).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
