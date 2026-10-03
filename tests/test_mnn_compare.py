import struct
import tempfile
import unittest
from pathlib import Path

import flatbuffers
from mnn_schema import AttentionParam, Blob, ExtraInfo, Net, Op, TensorQuantInfo
from mnn_schema.DataType import DataType
from mnn_schema.OpParameter import OpParameter
from mnn_schema.OpType import OpType

from src.sd15.compare_mnn import compare_models
from src.sd15.mnn_tensors import MnnModel


def write_model(
    path,
    values,
    biz_code="test",
    *,
    dimension=2,
    attention_scale=None,
    quantized_attention=False,
):
    builder = flatbuffers.Builder(256)
    version = builder.CreateString("3.6.1")
    name = builder.CreateString("embedding")
    biz = builder.CreateString(biz_code)
    data = builder.CreateByteVector(struct.pack("<2e", *values))
    Blob.BlobStartDimsVector(builder, 1)
    builder.PrependInt32(dimension)
    dims = builder.EndVector()
    Blob.BlobStart(builder)
    Blob.BlobAddDims(builder, dims)
    Blob.BlobAddDataType(builder, DataType.DT_HALF)
    Blob.BlobAddUint8s(builder, data)
    blob = Blob.BlobEnd(builder)
    Op.OpStart(builder)
    Op.OpAddName(builder, name)
    Op.OpAddMainType(builder, OpParameter.Blob)
    Op.OpAddMain(builder, blob)
    Op.OpAddType(builder, OpType.Const)
    op = Op.OpEnd(builder)
    attention = None
    if attention_scale is not None:
        attention_name = builder.CreateString("Attention/test")
        quant = None
        if quantized_attention:
            TensorQuantInfo.TensorQuantInfoStart(builder)
            info = TensorQuantInfo.TensorQuantInfoEnd(builder)
            AttentionParam.AttentionParamStartMhqQuantVector(builder, 1)
            builder.PrependUOffsetTRelative(info)
            quant = builder.EndVector()
        AttentionParam.AttentionParamStart(builder)
        AttentionParam.AttentionParamAddKvCache(builder, False)
        AttentionParam.AttentionParamAddAttnScale(builder, attention_scale)
        if quant is not None:
            AttentionParam.AttentionParamAddMhqQuant(builder, quant)
        param = AttentionParam.AttentionParamEnd(builder)
        Op.OpStart(builder)
        Op.OpAddName(builder, attention_name)
        Op.OpAddMainType(builder, OpParameter.AttentionParam)
        Op.OpAddMain(builder, param)
        Op.OpAddType(builder, OpType.Attention)
        attention = Op.OpEnd(builder)
    Net.NetStartOplistsVector(builder, 2 if attention is not None else 1)
    if attention is not None:
        builder.PrependUOffsetTRelative(attention)
    builder.PrependUOffsetTRelative(op)
    ops = builder.EndVector()
    ExtraInfo.ExtraInfoStart(builder)
    ExtraInfo.ExtraInfoAddVersion(builder, version)
    extra = ExtraInfo.ExtraInfoEnd(builder)
    Net.NetStart(builder)
    Net.NetAddBizCode(builder, biz)
    Net.NetAddExtraInfo(builder, extra)
    Net.NetAddOplists(builder, ops)
    builder.Finish(Net.NetEnd(builder))
    path.write_bytes(builder.Output())


class MnnComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "a").mkdir()
        (root / "b").mkdir()
        self.reference = root / "a/model.mnn"
        self.candidate = root / "b/model.mnn"

    def test_reads_tensors_independently_without_manifest(self):
        write_model(self.reference, [1, 2])
        write_model(self.candidate, [1, 2])
        self.assertTrue(compare_models(self.reference, self.candidate)["equal"])

    def test_reports_changed_tensor_and_element(self):
        write_model(self.reference, [1, 2])
        write_model(self.candidate, [1, 3])
        result = compare_models(self.reference, self.candidate)
        self.assertFalse(result["equal"])
        self.assertTrue(result["graph_equal"])
        self.assertEqual(result["tensors"][0]["op"], "embedding")
        self.assertEqual(result["tensors"][0]["first_element"], 1)
        self.assertEqual(result["tensors"][0]["max_abs_error"], 1)

    def test_signed_zero_is_not_hidden_by_tolerance(self):
        write_model(self.reference, [1, -0.0])
        write_model(self.candidate, [1, 0.0])
        result = compare_models(self.reference, self.candidate)
        self.assertFalse(result["equal"])
        self.assertEqual(result["tensors"][0]["max_abs_error"], 0)

    def test_graph_metadata_is_also_checked(self):
        write_model(self.reference, [1, 2], "left")
        write_model(self.candidate, [1, 2], "diff")
        result = compare_models(self.reference, self.candidate)
        self.assertTrue(result["tensors"][0]["equal"])
        self.assertFalse(result["equal"])

    def test_attention_settings_are_compared_as_graph_bytes(self):
        write_model(self.reference, [1, 2], attention_scale=0.125)
        write_model(self.candidate, [1, 2], attention_scale=0.125)
        self.assertTrue(compare_models(self.reference, self.candidate)["equal"])
        write_model(self.candidate, [1, 2], attention_scale=0.25)
        result = compare_models(self.reference, self.candidate)
        self.assertTrue(result["tensors"][0]["equal"])
        self.assertFalse(result["graph_equal"])

    def test_quantized_attention_is_rejected(self):
        write_model(
            self.reference, [1, 2], attention_scale=0.125, quantized_attention=True
        )
        with self.assertRaisesRegex(ValueError, "quantized Attention"):
            with MnnModel(self.reference):
                pass

    def test_invalid_shape_preserves_error_and_closes_resources(self):
        write_model(self.reference, [1, 2], dimension=3)
        # 생성 실패 후에도 자원 해제 여부를 확인할 수 있게 참조를 유지한다.
        model = MnnModel.__new__(MnnModel)
        with self.assertRaisesRegex(ValueError, "shape and byte length disagree"):
            model.__init__(self.reference)
        self.assertTrue(all(buffer.closed for buffer in model.buffers.values()))
        self.assertTrue(all(stream.closed for stream in model.files.values()))


if __name__ == "__main__":
    unittest.main()
