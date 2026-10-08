"""MNN 3.6.1의 직렬화된 텐서 구간을 읽는다. 모델 실행은 하지 않는다."""

import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mnn_schema.AttentionParam import AttentionParam
from mnn_schema.Blob import Blob
from mnn_schema.Convolution2D import Convolution2D
from mnn_schema.DataType import DataType
from mnn_schema.LayerNorm import LayerNorm
from mnn_schema.MatMul import MatMul
from mnn_schema.Net import Net
from mnn_schema.OpParameter import OpParameter

from .template_values import MNN_VERSION

FP16_QUANT_TYPE = 3
DENSE_QUANT_TYPE = 1
W8_BITS = 8
W8_VALUE_COUNT = 1 << W8_BITS


@dataclass
class TensorSlot:
    """연산·필드에 속한 텐서의 파일 내 바이트 구간과 저장 shape·dtype."""

    op: str
    field: str
    file: str
    offset: int
    size: int
    dtype: str
    shape: list[int]
    quantization: dict | None = None


class MnnModel:
    """MNN과 외부 가중치를 읽기 전용 mmap으로 열어 텐서 구간을 노출한다.

    컨텍스트 매니저로 사용하며 종료 전에 파생 memoryview를 해제해야 한다.
    """

    def __init__(self, path):
        """MNN 3.6.1 파일을 열고 지원하는 저장 구조의 텐서를 수집한다."""
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"MNN model not found: {self.path}")
        self.files = {}
        self.buffers = {}
        try:
            for candidate in (self.path, Path(str(self.path) + ".weight")):
                if candidate.is_file():
                    stream = candidate.open("rb")
                    self.files[candidate.name] = stream
                    self.buffers[candidate.name] = mmap.mmap(
                        stream.fileno(), 0, access=mmap.ACCESS_READ
                    )
            self.net = Net.GetRootAs(self.buffers[self.path.name], 0)
            if self.net.ExtraInfo().Version().decode() != MNN_VERSION:
                raise ValueError(f"Expected MNN {MNN_VERSION}: {path}")
            if self.net.SubgraphsLength():
                raise ValueError("Subgraphs are not supported by this SD1.5 reader")
            self.slots = self._slots()
        except Exception:
            self.close()
            raise

    def close(self):
        """모든 mmap과 파일 핸들을 해제한다."""
        for buffer in self.buffers.values():
            buffer.close()
        for stream in self.files.values():
            stream.close()

    def __enter__(self):
        """열린 모델을 컨텍스트에 제공한다."""
        return self

    def __exit__(self, *_):
        """정상 종료와 예외 발생 모두에서 모델 자원을 해제한다."""
        self.close()

    def _slots(self):
        """스키마에서 텐서 위치를 추출하고 지원하지 않는 저장 구조는 거부한다."""
        slots = []
        buffer = self.buffers[self.path.name]
        base = np.frombuffer(buffer, dtype="u1").ctypes.data

        def add(op, field, file, offset, size, dtype, shape):
            """비어 있지 않은 텐서의 크기와 파일 범위를 검증해 등록한다."""
            if not size:
                return
            widths = {"F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1}
            if size != int(np.prod(shape, dtype=object)) * widths[dtype]:
                raise ValueError(f"{op}/{field}: shape and byte length disagree")
            if (
                file not in self.buffers
                or not 0 <= offset <= len(self.buffers[file]) - size
            ):
                raise ValueError(f"{op}/{field}: payload outside {file}")
            slots.append(TensorSlot(op, field, file, offset, size, dtype, shape))

        def inline(op, obj, field, dtype, shape):
            """FlatBuffer 내부 배열의 주소를 모델 파일 offset으로 바꿔 등록한다."""
            array = getattr(obj, field + "AsNumpy")()
            try:
                if isinstance(array, np.ndarray) and array.size:
                    add(
                        op,
                        field,
                        self.path.name,
                        array.ctypes.data - base,
                        array.nbytes,
                        dtype,
                        shape,
                    )
            finally:
                # 검증 예외의 traceback이 mmap 참조를 붙잡지 않게 한다.
                del array

        def dense(op, conv, quant, shape):
            """dense W8의 고정 헤더를 검증하고 payload·alpha 구간을 등록한다."""
            if (
                quant.AMaxOrBits() != W8_BITS
                or quant.AMin() != -128
                or quant.UseInt32()
                or quant.HasScaleInt()
                or quant.ScaleStorage() != 0
                or quant.AlphaFp16Length()
            ):
                raise ValueError(
                    f"{op}: unsupported dense W8 configuration "
                    f"(bits={quant.AMaxOrBits()}, aMin={quant.AMin()}, "
                    f"readType={quant.ReadType()}, scaleStorage={quant.ScaleStorage()}, "
                    f"alphaFp16={quant.AlphaFp16Length()}); "
                    "expected asymmetric W8 with FP32 min/scale. "
                    "Prepare the reference with --weightQuantAsymmetric=1 --hqq"
                )
            array = quant.BufferAsNumpy()
            try:
                if not isinstance(array, np.ndarray) or not array.size:
                    raise ValueError(f"{op}: missing W8 payload")
                offset = array.ctypes.data - base
                width = 4 if quant.ShapeInt32() else 2
                if array.size < 1 + 2 * width + 1 + W8_VALUE_COUNT:
                    raise ValueError(f"{op}: truncated W8 header")
                header = buffer[offset : offset + 1 + 2 * width + 1 + W8_VALUE_COUNT]
                if header[0] != 2:
                    raise ValueError(f"{op}: expected two W8 dimensions")
                groups, area = struct.unpack_from(
                    "<2I" if width == 4 else "<2H", header, 1
                )
                table = bytes(range(128, 256)) + bytes(range(128))
                if header[1 + 2 * width] != 0 or header[2 + 2 * width :] != table:
                    raise ValueError(f"{op}: unsupported W8 value table")
                if not groups or not area or groups * area != math.prod(shape):
                    raise ValueError(f"{op}: W8 shape mismatch")
                if quant.ReadType() != groups or quant.AlphaLength() != groups * 2:
                    raise ValueError(f"{op}: W8 alpha/group mismatch")
                if array.size != len(header) + groups * area:
                    raise ValueError(f"{op}: W8 payload length mismatch")
                info = {"bits": W8_BITS, "group_elements": area, "group_count": groups}
                add(
                    op,
                    "Weight",
                    self.path.name,
                    offset + len(header),
                    groups * area,
                    "U8",
                    shape,
                )
                slots[-1].quantization = info
            finally:
                del array
            inline(op, quant, "Alpha", "F32", [groups, 2])
            slots[-1].quantization = info
            inline(op, conv, "Bias", "F32", [shape[0]])

        parameterless = {
            "NONE",
            "Reshape",
            "BinaryOp",
            "TensorConvertInfo",
            "Axis",
            "SqueezeParam",
            "Permute",
            "UnaryOp",
            "Input",
            "GatherV2",
            "StridedSliceParam",
            "Interp",
            "CastParam",
        }
        names = {v: k for k, v in vars(OpParameter).items() if isinstance(v, int)}
        for i in range(self.net.OplistsLength()):
            op = self.net.Oplists(i)
            name = op.Name().decode()
            if op.ExternalPath():
                raise ValueError(f"{name}: per-op external paths are unsupported")
            kind = names[op.MainType()]
            if kind in parameterless:
                continue
            if kind == "AttentionParam":
                attention = AttentionParam()
                attention.Init(op.Main().Bytes, op.Main().Pos)
                if attention.MhqQuantLength():
                    raise ValueError(f"{name}: quantized Attention is unsupported")
                # Q/K/V는 입력 텐서다. Attention 설정은 고정 그래프 바이트로 보존한다.
                continue
            classes = {
                "Blob": Blob,
                "Convolution2D": Convolution2D,
                "LayerNorm": LayerNorm,
                "MatMul": MatMul,
            }
            if kind not in classes:
                raise ValueError(f"Unhandled parameter type: {name}: {kind}")
            obj = classes[kind]()
            obj.Init(op.Main().Bytes, op.Main().Pos)
            ext = (
                []
                if kind == "MatMul"
                else [obj.External(j) for j in range(obj.ExternalLength())]
            )
            external_file = self.path.name + ".weight"
            if kind == "Blob":
                shape = [obj.Dims(j) for j in range(obj.DimsLength())]
                dtype, field = {
                    DataType.DT_FLOAT: ("F32", "Float32s"),
                    DataType.DT_INT32: ("I32", "Int32s"),
                    DataType.DT_INT64: ("I64", "Int64s"),
                    DataType.DT_HALF: ("F16", "Uint8s"),
                }[obj.DataType()]
                if ext:
                    if len(ext) != 2:
                        raise ValueError(f"{name}: unsupported Blob external layout")
                    add(name, field, external_file, ext[0], ext[1], dtype, shape)
                else:
                    inline(name, obj, field, dtype, shape)
            elif kind == "Convolution2D":
                common = obj.Common()
                shape = [
                    common.OutputCount(),
                    common.InputCount() // common.Group(),
                    common.KernelY(),
                    common.KernelX(),
                ]
                quant = obj.QuanParameter()
                if quant is not None and quant.Type() == DENSE_QUANT_TYPE:
                    if ext:
                        raise ValueError(f"{name}: external dense W8 is unsupported")
                    dense(name, obj, quant, shape)
                    continue
                if quant is None or quant.Type() != FP16_QUANT_TYPE:
                    raise ValueError(f"{name}: expected FP16 convolution storage")
                if ext:
                    if len(ext) != 5 or ext[2] != 0 or ext[4] != 0:
                        raise ValueError(
                            f"{name}: unsupported quantized external layout"
                        )
                    add(name, "Weight", external_file, ext[0], ext[1], "F16", shape)
                    add(
                        name,
                        "Bias",
                        external_file,
                        ext[0] + ext[1],
                        ext[3],
                        "F32",
                        [shape[0]],
                    )
                else:
                    start = len(slots)
                    inline(name, quant, "Buffer", "F16", shape)
                    if len(slots) != start + 1:
                        raise ValueError(f"{name}: missing convolution weight")
                    slots[-1].field = "Weight"
                    inline(name, obj, "Bias", "F32", [shape[0]])
            elif kind == "LayerNorm":
                if ext:
                    if len(ext) != 3:
                        raise ValueError(
                            f"{name}: unsupported LayerNorm external layout"
                        )
                    add(
                        name,
                        "Gamma",
                        external_file,
                        ext[0],
                        ext[1],
                        "F32",
                        [ext[1] // 4],
                    )
                    add(
                        name,
                        "Beta",
                        external_file,
                        ext[0] + ext[1],
                        ext[2],
                        "F32",
                        [ext[2] // 4],
                    )
                else:
                    inline(name, obj, "Gamma", "F32", [obj.GammaLength()])
                    inline(name, obj, "Beta", "F32", [obj.BetaLength()])
            elif obj.WeightLength() or obj.BiasLength():
                raise ValueError(f"{name}: embedded MatMul parameters are unsupported")
        return slots
