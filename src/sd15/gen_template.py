"""기준 SD1.5 safetensors/ONNX/MNN에서 복원용 템플릿을 추출한다."""

import argparse
import gc
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from safetensors import safe_open

from .mnn_tensors import MnnModel
from .template_values import (
    CHUNK_BYTES,
    COMPONENTS,
    FORMAT_NAME,
    FORMAT_VERSION,
    MNN_VERSION,
    encode_tensor,
)


def digest(data):
    """텐서 저장 바이트의 SHA-256을 반환한다."""
    return hashlib.sha256(data).hexdigest()


def index_source(checkpoint):
    """원본의 ONNX 변형·MNN 저장값 해시 인덱스와 shape 표를 만든다.

    PC용 분석이며 텐서 전체를 메모리에 읽는다. F32/F16만 색인한다.
    """
    onnx_index = defaultdict(list)
    stored_index = defaultdict(set)
    shapes = {}
    with safe_open(checkpoint, framework="numpy") as source:
        for key in source.keys():
            info = source.get_slice(key)
            if info.get_dtype() not in ("F32", "F16"):
                continue
            values = source.get_tensor(key).astype("<f4", copy=False)
            shapes[key] = list(values.shape)
            onnx_index[digest(values.tobytes())].append((key, "reshape"))
            squeezed = values.squeeze()
            if squeezed.ndim == 2:
                onnx_index[digest(squeezed.T.tobytes())].append(
                    (key, "squeeze_transpose")
                )
            for dtype in ("F32", "F16"):
                for positive_zero in (False, True):
                    data = encode_tensor(values, dtype, positive_zero)
                    stored_index[(dtype, len(data), digest(data), positive_zero)].add(
                        key
                    )
    return onnx_index, stored_index, shapes


def onnx_sources(path, index):
    """initializer를 원본 key에 연결하고 그래프 입출력 정보를 반환한다.

    연결 후보가 하나가 아니거나 원본 key가 중복되면 실패한다.
    """
    model = onnx.load(path, load_external_data=False)
    sources = {}
    for tensor in model.graph.initializer:
        values = onnx.numpy_helper.to_array(tensor, base_dir=str(path.parent))
        candidates = index.get(digest(values.tobytes()), [])
        if len(candidates) != 1:
            raise ValueError(f"{tensor.name}: expected one source, got {candidates}")
        key, transform = candidates[0]
        if key in sources:
            raise ValueError(f"{key}: duplicate ONNX parameter; provenance required")
        sources[key] = {
            "name": tensor.name,
            "shape": list(tensor.dims),
            "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
            "transform": transform,
        }
    inputs = [onnx.helper.printable_value_info(t) for t in model.graph.input]
    outputs = [onnx.helper.printable_value_info(t) for t in model.graph.output]
    return sources, {"inputs": inputs, "outputs": outputs}


def classify(model, sources, index, shapes):
    """MNN 구간을 원본 가중치와 0 상수로 분류하고 매핑 누락을 검사한다.

    고정 바이트는 분류에서 제외하며, 중복 후보와 미매핑 가중치는 거부한다.
    """
    regions = defaultdict(list)
    covered = set()
    for slot in model.slots:
        buffer = model.buffers[slot.file]
        with memoryview(buffer)[slot.offset : slot.offset + slot.size] as data:
            slot_digest = digest(data)
        # MNN MatMul->Conv 변환은 원본의 -0을 +0으로 만든다.
        positive_zero = (
            slot.op.endswith("__matmul_converted") and slot.field == "Weight"
        )
        candidates = index.get(
            (slot.dtype, slot.size, slot_digest, positive_zero), set()
        )
        candidates = candidates.intersection(sources)
        region = {
            "offset": slot.offset,
            "size": slot.size,
            "op": slot.op,
            "field": slot.field,
            "dtype": slot.dtype,
            "shape": slot.shape,
        }
        if len(candidates) > 1:
            raise ValueError(f"{slot.op}/{slot.field}: ambiguous sources {candidates}")
        if candidates:
            key = next(iter(candidates))
            covered.add(key)
            region.update(
                kind="tensor",
                key=key,
                source_shape=shapes[key],
                positive_zero=positive_zero,
                onnx=sources[key],
            )
        elif not np.frombuffer(
            buffer, dtype="u1", offset=slot.offset, count=slot.size
        ).any():
            if slot.field not in ("Uint8s", "Int32s", "Bias"):
                raise ValueError(f"Unmapped zero parameter: {slot.op}/{slot.field}")
            region["kind"] = "zero"
        elif slot.field in ("Weight", "Bias", "Gamma", "Beta"):
            raise ValueError(f"Unmapped parameter: {slot.op}/{slot.field}")
        else:
            # 고정 상수는 원본 바이트를 보관한다.
            continue
        regions[slot.file].append(region)
    if covered != set(sources):
        raise ValueError(f"Missing MNN parameters: {sorted(set(sources) - covered)}")
    return regions


def write_component(model, regions, template):
    """고정 바이트를 템플릿에 추가하고 원래 파일을 복원할 구간 표를 반환한다."""
    outputs = []
    for name, buffer in model.buffers.items():
        segments = []
        cursor = 0
        for region in sorted(regions[name], key=lambda r: r["offset"]) + [
            {"offset": len(buffer), "size": 0}
        ]:
            offset = region["offset"]
            if offset < cursor:
                raise ValueError(f"{name}: overlapping tensor ranges")
            if offset > cursor:
                segments.append(
                    {
                        "kind": "literal",
                        "offset": cursor,
                        "size": offset - cursor,
                        "template_offset": template.tell(),
                    }
                )
                for start in range(cursor, offset, CHUNK_BYTES):
                    template.write(buffer[start : min(start + CHUNK_BYTES, offset)])
            if region["size"]:
                segments.append(region)
            cursor = offset + region["size"]
        outputs.append({"name": name, "size": len(buffer), "segments": segments})
    return outputs


def generate(checkpoint, onnx_dir, mnn_dir, output):
    """기준 모델에서 graph.bin과 manifest.json을 새 디렉터리에 생성한다.

    기존 출력은 덮어쓰지 않으며 실패 시 이번 생성물만 제거한다.
    """
    output.mkdir(parents=True, exist_ok=False)
    try:
        print("Indexing reference safetensors", flush=True)
        onnx_index, stored_index, shapes = index_source(str(checkpoint))
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "mnn_version": MNN_VERSION,
            "template": "graph.bin",
            "files": [],
            "components": {},
        }
        with (output / "graph.bin").open("xb") as template:
            for component in COMPONENTS:
                sources, io = onnx_sources(
                    onnx_dir / component / "model.onnx", onnx_index
                )
                gc.collect()
                with MnnModel(mnn_dir / f"{component}.mnn") as model:
                    regions = classify(model, sources, stored_index, shapes)
                    manifest["files"].extend(write_component(model, regions, template))
                manifest["components"][component] = io | {"tensor_count": len(sources)}
                print(f"{component}: {len(sources)} tensors mapped", flush=True)
            manifest["template_size"] = template.tell()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Template: {manifest['template_size']:,} bytes; saved {output}", flush=True
        )
    except Exception:
        shutil.rmtree(output)
        raise


def main():
    """CLI 경로 인자를 읽어 SD1.5 템플릿을 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/v1-5-pruned-emaonly.safetensors"),
    )
    parser.add_argument("--onnx-dir", type=Path, default=Path("/artifacts/onnx"))
    parser.add_argument("--mnn-dir", type=Path, default=Path("/artifacts/mnn"))
    parser.add_argument(
        "--output", type=Path, default=Path("/artifacts/templates/sd15")
    )
    args = parser.parse_args()
    generate(args.checkpoint, args.onnx_dir, args.mnn_dir, args.output)


if __name__ == "__main__":
    main()
