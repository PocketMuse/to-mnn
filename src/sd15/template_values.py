"""MNN 3.6.1 가중치 저장 규칙. 런타임 연산 정밀도와는 별개다."""

import numpy as np

MNN_VERSION = "3.6.1"
FORMAT_VERSION = 1
FORMAT_NAME = "sd15-mnn-template"
COMPONENTS = ("text_encoder", "unet", "vae_decoder")
CHUNK_BYTES = 1024 * 1024
HALF_MAX = 65504.0


def encode_tensor(values, dtype, positive_zero):
    """유한 가중치를 MNN의 F32/F16 바이트로 인코딩한다.

    F16은 범위를 제한하고 0 방향으로 절삭한다. positive_zero는 입력의
    signed zero만 정규화하며, 절삭으로 생긴 음의 0은 유지한다.
    """
    values = np.asarray(values, dtype="<f4").reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite learned weight")
    if positive_zero:
        values = values.copy()
        values[values == 0] = 0
    if dtype == "F32":
        return values.tobytes(order="C")
    if dtype != "F16":
        raise ValueError(f"Unsupported storage dtype: {dtype}")

    values = np.clip(values, -HALF_MAX, HALF_MAX)
    halves = values.astype("<f2")
    # NumPy는 nearest-even, MNN half.hpp 기본값은 toward-zero다.
    bits = halves.view("<u2")
    bits[np.abs(halves.astype("<f4")) > np.abs(values)] -= 1
    return halves.tobytes(order="C")
