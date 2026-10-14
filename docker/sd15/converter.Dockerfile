FROM debian:bookworm-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make cmake ninja-build python3 git curl ca-certificates zip unzip tar pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV VCPKG_ROOT=/opt/vcpkg
COPY cpp/vcpkg.json cpp/vcpkg.json
RUN git init "$VCPKG_ROOT" \
    && git -C "$VCPKG_ROOT" remote add origin https://github.com/microsoft/vcpkg.git \
    && git -C "$VCPKG_ROOT" fetch --depth 1 origin "$(python3 -c 'import json; print(json.load(open("cpp/vcpkg.json"))["builtin-baseline"])')" \
    && git -C "$VCPKG_ROOT" checkout --detach FETCH_HEAD \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
COPY cpp cpp
COPY tests tests
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build build -j2 \
    && ctest --test-dir build --output-on-failure

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 time \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /workspace/build/cli/sd15-convert /usr/local/bin/sd15-convert
COPY --from=build /workspace/build/converter/sd15-api-test /usr/local/bin/sd15-api-test
ENTRYPOINT ["sd15-convert"]
