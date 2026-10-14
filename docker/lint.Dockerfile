FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-format-14=1:14.0.6-12 clang-tidy-14=1:14.0.6-12 \
        g++ cmake make ninja-build python3 git curl ca-certificates zip unzip tar pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV VCPKG_ROOT=/opt/vcpkg
COPY cpp/vcpkg.json /opt/manifest/vcpkg.json
RUN git init "$VCPKG_ROOT" \
    && git -C "$VCPKG_ROOT" remote add origin https://github.com/microsoft/vcpkg.git \
    && git -C "$VCPKG_ROOT" fetch --depth 1 origin "$(python3 -c 'import json; print(json.load(open("/opt/manifest/vcpkg.json"))["builtin-baseline"])')" \
    && git -C "$VCPKG_ROOT" checkout --detach FETCH_HEAD \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics \
    && "$VCPKG_ROOT/vcpkg" install --x-feature=cli --x-manifest-root=/opt/manifest --x-install-root=/opt/vcpkg_installed

WORKDIR /workspace
ENTRYPOINT ["bash", "scripts/lint-cpp.sh"]
CMD ["format-check"]
