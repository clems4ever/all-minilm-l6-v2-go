FROM golang:1.25 AS build

# Install dependencies for downloading and extracting ONNX Runtime
RUN apt-get update && apt-get install -y wget

# Set up work directory
WORKDIR /go/src/app

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Install ONNX Runtime
ARG ORT_VERSION=1.23.0
RUN mkdir -p /usr/local/include /usr/local/lib && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    cp -r onnxruntime-linux-x64-${ORT_VERSION}/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-${ORT_VERSION}/lib/* /usr/local/lib/ && \
    rm -rf onnxruntime-linux-x64-${ORT_VERSION}*

# Copy source code
COPY cmd cmd
COPY all_minilm_l6_v2 all_minilm_l6_v2

# Build the application
RUN CGO_ENABLED=1 go build -o app ./cmd/cli

# Runtime stage - use a minimal glibc-based image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime libraries from build stage
COPY --from=build /usr/local/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=build /go/src/app/app /app

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV ONNXRUNTIME_LIB_PATH=/usr/local/lib/libonnxruntime.so

ENTRYPOINT ["/app"]