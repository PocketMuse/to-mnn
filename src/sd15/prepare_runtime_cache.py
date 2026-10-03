"""Docker에서 생성한 설정·tokenizer 캐시를 Windows에서도 읽도록 준비한다."""

import os
import shutil
import tempfile
from pathlib import Path

DEFAULT_CACHE = Path(
    "/artifacts/huggingface/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
)


def prepare_runtime_cache(cache=DEFAULT_CACHE):
    """현재 snapshot의 파일 링크를 실제 파일로 교체하고 blob은 보존한다."""
    revision = (cache / "refs/main").read_text(encoding="utf-8").strip()
    snapshot = cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)

    count = 0
    for path in snapshot.rglob("*"):
        if not path.is_symlink():
            continue
        # 복사를 완료한 뒤 링크만 교체한다. 실패하면 원래 링크를 유지한다.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(path, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        count += 1
    return count


if __name__ == "__main__":
    print(f"Prepared runtime cache: {prepare_runtime_cache()} links materialized")
