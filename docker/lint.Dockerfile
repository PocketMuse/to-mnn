FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-format-14=1:14.0.6-12 clang-tidy-14=1:14.0.6-12 \
        g++ cmake make nlohmann-json3-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENTRYPOINT ["bash", "scripts/lint-cpp.sh"]
CMD ["format-check"]
