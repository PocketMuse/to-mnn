FROM debian:bookworm-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make cmake nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY cpp cpp
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build build -j2

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 time \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /workspace/build/sd15-convert /usr/local/bin/sd15-convert
ENTRYPOINT ["sd15-convert"]
