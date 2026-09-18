FROM gcc:15-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TESTING=OFF \
    && cmake --build build --parallel \
    && mkdir -p /opt/microtensor/include/microtensor /opt/microtensor/lib \
    && cp src/microtensor/*.hpp /opt/microtensor/include/microtensor/ \
    && cp build/src/microtensor/libmicrotensor_lib.a /opt/microtensor/lib/

FROM gcc:15-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/microtensor /opt/microtensor

ENV CPLUS_INCLUDE_PATH=/opt/microtensor/include
ENV LIBRARY_PATH=/opt/microtensor/lib

WORKDIR /workspace

LABEL org.opencontainers.image.source="https://github.com/BhagyeshKothalkar/microtensor"
