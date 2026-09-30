"""manifest의 offset에 의존하지 않고 두 MNN의 텐서를 직접 비교한다."""

import argparse
import json
from pathlib import Path

import numpy as np

from .mnn_tensors import MnnModel
from .template_values import CHUNK_BYTES, COMPONENTS

DTYPES = {"F16": "<f2", "F32": "<f4", "I32": "<i4", "I64": "<i8"}


def compare_slot(reference, candidate, left, right):
    """shape·dtype·바이트를 비교해 첫 불일치와 최대 절대 오차를 반환한다.

    허용 오차 없이 signed zero를 포함한 저장 바이트 일치를 판정한다.
    """
    result = {
        "op": left.op,
        "field": left.field,
        "shape": left.shape,
        "dtype": left.dtype,
        "bytes": left.size,
        "equal": True,
    }
    if (left.shape, left.dtype, left.size) != (right.shape, right.dtype, right.size):
        return result | {
            "equal": False,
            "reason": "shape/dtype/length mismatch",
            "actual_shape": right.shape,
            "actual_dtype": right.dtype,
            "actual_bytes": right.size,
        }
    dtype = np.dtype(DTYPES[left.dtype])
    max_error = 0.0
    for cursor in range(0, left.size, CHUNK_BYTES):
        size = min(CHUNK_BYTES, left.size - cursor)
        a = reference.buffers[left.file][
            left.offset + cursor : left.offset + cursor + size
        ]
        b = candidate.buffers[right.file][
            right.offset + cursor : right.offset + cursor + size
        ]
        if a == b:
            continue
        expected = np.frombuffer(a, dtype=dtype)
        actual = np.frombuffer(b, dtype=dtype)
        if result["equal"]:
            byte_index = next(
                i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y
            )
            index = byte_index // dtype.itemsize
            result.update(
                equal=False,
                reason="tensor bytes differ",
                first_element=(cursor // dtype.itemsize) + index,
                expected=float(expected[index]),
                actual=float(actual[index]),
                expected_hex=a[
                    index * dtype.itemsize : (index + 1) * dtype.itemsize
                ].hex(),
                actual_hex=b[
                    index * dtype.itemsize : (index + 1) * dtype.itemsize
                ].hex(),
            )
        max_error = max(
            max_error,
            float(np.max(np.abs(expected.astype("f8") - actual.astype("f8")))),
        )
    if not result["equal"]:
        result["max_abs_error"] = max_error
    return result


def graph_bytes_equal(reference, candidate):
    """파일 구성·텐서 배치와 텐서 외 구간의 바이트가 같은지 검사한다."""
    if set(reference.buffers) != set(candidate.buffers):
        return False
    for name, buffer in reference.buffers.items():
        other = candidate.buffers[name]
        if len(buffer) != len(other):
            return False
        ranges = sorted((s.offset, s.size) for s in reference.slots if s.file == name)
        other_ranges = sorted(
            (s.offset, s.size) for s in candidate.slots if s.file == name
        )
        if ranges != other_ranges:
            return False
        cursor = 0
        for offset, size in ranges + [(len(buffer), 0)]:
            for start in range(cursor, offset, CHUNK_BYTES):
                end = min(start + CHUNK_BYTES, offset)
                if buffer[start:end] != other[start:end]:
                    return False
            cursor = offset + size
    return True


def compare_models(reference_path, candidate_path):
    """두 MNN을 독립 파싱하여 텐서별 결과와 그래프 일치 여부를 반환한다.

    연산 이름·필드로 텐서를 대응시키며 중복이나 텐서 집합 차이는 거부한다.
    """
    with MnnModel(reference_path) as reference, MnnModel(candidate_path) as candidate:
        left = {(s.op, s.field): s for s in reference.slots}
        right = {(s.op, s.field): s for s in candidate.slots}
        if len(left) != len(reference.slots) or len(right) != len(candidate.slots):
            raise ValueError("Duplicate op/field tensor identity")
        if left.keys() != right.keys():
            raise ValueError(
                f"Tensor set mismatch: missing={left.keys() - right.keys()}, extra={right.keys() - left.keys()}"
            )
        tensors = [
            compare_slot(reference, candidate, slot, right[key])
            for key, slot in left.items()
        ]
        graph_equal = graph_bytes_equal(reference, candidate)
        return {
            "tensors": tensors,
            "graph_equal": graph_equal,
            "equal": all(t["equal"] for t in tensors) and graph_equal,
        }


def main():
    """구성 요소별 비교 보고서를 기록하고 불일치 시 종료 코드 1을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("/artifacts/mnn"))
    parser.add_argument("--candidate", type=Path, default=Path("/artifacts/mnn-cpp"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("/artifacts/tensor-comparison.json")
    )
    args = parser.parse_args()
    labels = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        labels = {
            (f["name"].removesuffix(".weight"), s["op"], s["field"]): s["key"]
            for f in manifest["files"]
            for s in f["segments"]
            if s["kind"] == "tensor"
        }
    report = {}
    for name in COMPONENTS:
        filename = name + ".mnn"
        result = compare_models(args.reference / filename, args.candidate / filename)
        for tensor in result["tensors"]:
            tensor["source_key"] = labels.get((filename, tensor["op"], tensor["field"]))
        report[name] = result
        count = sum(t["equal"] for t in result["tensors"])
        print(
            f"{name}: {count}/{len(result['tensors'])} tensors exact, graph={result['graph_equal']}",
            flush=True,
        )
        for tensor in result["tensors"]:
            if not tensor["equal"]:
                print(json.dumps(tensor), flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["equal"] for r in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
