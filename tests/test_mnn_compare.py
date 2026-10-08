import struct
import tempfile
import unittest
from pathlib import Path

import flatbuffers
from mnn_schema import (
    AttentionParam,
    Blob,
    Convolution2D,
    Convolution2DCommon,
    ExtraInfo,
    IDSTQuan,
    Net,
    Op,
    TensorQuantInfo,
)
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


def write_quant_model(
    path,
    payload=b"\x00\xff",
    alpha=(-1.0, 2 / 255),
    *,
    bits=8,
    wide=False,
    minimum=-128,
):
    """두 원소의 dense W8 convolution을 만든다."""
    builder = flatbuffers.Builder(512)
    version = builder.CreateString("3.6.1")
    name = builder.CreateString("conv")
    header = bytes([2]) + struct.pack("<2I" if wide else "<2H", 1, 2)
    header += bytes([0]) + bytes(range(128, 256)) + bytes(range(128))
    buffer = builder.CreateByteVector(header + payload)
    IDSTQuan.IDSTQuanStartAlphaVector(builder, len(alpha))
    for value in reversed(alpha):
        builder.PrependFloat32(value)
    scales = builder.EndVector()
    IDSTQuan.IDSTQuanStart(builder)
    IDSTQuan.IDSTQuanAddBuffer(builder, buffer)
    IDSTQuan.IDSTQuanAddAlpha(builder, scales)
    IDSTQuan.IDSTQuanAddType(builder, 1)
    IDSTQuan.IDSTQuanAddAMaxOrBits(builder, bits)
    IDSTQuan.IDSTQuanAddAMin(builder, minimum)
    IDSTQuan.IDSTQuanAddReadType(builder, 1)
    IDSTQuan.IDSTQuanAddShapeInt32(builder, wide)
    quant = IDSTQuan.IDSTQuanEnd(builder)
    Convolution2DCommon.Convolution2DCommonStart(builder)
    Convolution2DCommon.Convolution2DCommonAddInputCount(builder, 2)
    Convolution2DCommon.Convolution2DCommonAddOutputCount(builder, 1)
    common = Convolution2DCommon.Convolution2DCommonEnd(builder)
    Convolution2D.Convolution2DStart(builder)
    Convolution2D.Convolution2DAddCommon(builder, common)
    Convolution2D.Convolution2DAddQuanParameter(builder, quant)
    conv = Convolution2D.Convolution2DEnd(builder)
    Op.OpStart(builder)
    Op.OpAddName(builder, name)
    Op.OpAddMainType(builder, OpParameter.Convolution2D)
    Op.OpAddMain(builder, conv)
    Op.OpAddType(builder, OpType.Convolution)
    op = Op.OpEnd(builder)
    Net.NetStartOplistsVector(builder, 1)
    builder.PrependUOffsetTRelative(op)
    ops = builder.EndVector()
    ExtraInfo.ExtraInfoStart(builder)
    ExtraInfo.ExtraInfoAddVersion(builder, version)
    extra = ExtraInfo.ExtraInfoEnd(builder)
    Net.NetStart(builder)
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

    def test_missing_model_reports_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "MNN model not found") as error:
            MnnModel(self.reference)
        self.assertIn(str(self.reference), str(error.exception))

    def test_reads_tensors_independently_without_manifest(self):
        write_model(self.reference, [1, 2])
        write_model(self.candidate, [1, 2])
        self.assertTrue(compare_models(self.reference, self.candidate)["equal"])

    def test_symmetric_w8_reports_reference_preparation_option(self):
        write_quant_model(self.reference, minimum=0)
        with self.assertRaisesRegex(ValueError, "aMin=0.*weightQuantAsymmetric=1"):
            MnnModel(self.reference)

    def test_dense_w8_payload_and_alpha_are_compared(self):
        write_quant_model(self.reference)
        write_quant_model(self.candidate, alpha=(-0.5, 2 / 255))
        result = compare_models(self.reference, self.candidate)
        self.assertTrue(result["graph_equal"])
        self.assertEqual({t["field"] for t in result["tensors"]}, {"Weight", "Alpha"})
        self.assertFalse(result["equal"])

    def test_dense_w8_supports_int32_header_dimensions(self):
        write_quant_model(self.reference, wide=True)
        with MnnModel(self.reference) as model:
            weight = next(s for s in model.slots if s.field == "Weight")
            self.assertEqual(weight.dtype, "U8")
            self.assertEqual(weight.quantization["group_elements"], 2)
            self.assertEqual(
                model.buffers[weight.file][weight.offset : weight.offset + 2],
                b"\x00\xff",
            )

    def test_dense_w8_rejects_wrong_alpha_length(self):
        write_quant_model(self.reference, alpha=(-1.0,))
        with self.assertRaisesRegex(ValueError, "alpha"):
            with MnnModel(self.reference):
                pass

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
