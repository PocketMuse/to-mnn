FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace

# CPU 실행 라이브러리와 HTTPS 인증서
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src/sd15/export_mnn.py /workspace/sd15/export_mnn.py

RUN uv sync --locked \
    --no-default-groups \
    --no-install-project

ENTRYPOINT ["python", "/workspace/sd15/export_mnn.py"]
