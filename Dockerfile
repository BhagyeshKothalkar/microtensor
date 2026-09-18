ARG CMAKE_VERSION=3.30.9
ARG CMAKE_SHA256=9114e33358a9efc93d6ea658805280fc3201b882b944a4d946edd9472fd1eec7

FROM gcc:15-bookworm AS toolchain

ARG CMAKE_VERSION
ARG CMAKE_SHA256

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ninja-build \
    && mkdir -p /opt/cmake \
    && curl -fsSL \
       "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
       -o /tmp/cmake.tar.gz \
    && echo "${CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/cmake.tar.gz --strip-components=1 -C /opt/cmake \
    && rm -f /tmp/cmake.tar.gz \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/cmake/bin:${PATH}"

FROM toolchain AS builder

WORKDIR /src
COPY . .

RUN cmake --version \
    && cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TESTING=OFF \
    && cmake --build build --parallel \
    && mkdir -p /opt/microtensor/include/microtensor /opt/microtensor/lib \
    && cp src/microtensor/*.hpp /opt/microtensor/include/microtensor/ \
    && cp build/src/microtensor/libmicrotensor_lib.a /opt/microtensor/lib/

FROM toolchain

COPY --from=builder /opt/microtensor /opt/microtensor

ENV CPLUS_INCLUDE_PATH=/opt/microtensor/include
ENV LIBRARY_PATH=/opt/microtensor/lib

WORKDIR /workspace

LABEL org.opencontainers.image.source="https://github.com/BhagyeshKothalkar/microtensor"
