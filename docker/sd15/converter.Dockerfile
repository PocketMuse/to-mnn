FROM debian:bookworm-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make cmake nlohmann-json3-dev python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY cpp cpp
COPY tests tests
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build build -j2 \
    && ctest --test-dir build --output-on-failure

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 time \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /workspace/build/sd15-convert /usr/local/bin/sd15-convert
COPY --from=build /workspace/build/sd15-api-test /usr/local/bin/sd15-api-test
ENTRYPOINT ["sd15-convert"]
