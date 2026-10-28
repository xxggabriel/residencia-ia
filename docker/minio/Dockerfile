# Official community source, fixed to the security release commit. AGPL-3.0.
FROM golang:1.24.8-bookworm AS build
ENV GOMAXPROCS=4 GOTOOLCHAIN=local
RUN git clone https://github.com/minio/minio.git /src \
 && cd /src && git checkout --detach 9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a \
 && test "$(git rev-parse HEAD)" = 9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a
WORKDIR /src
RUN CGO_ENABLED=0 go build -p 4 -trimpath \
    -ldflags "$(MINIO_RELEASE=RELEASE go run buildscripts/gen-ldflags.go 2025-10-15T17:29:55Z)" \
    -o /minio .
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /minio /usr/local/bin/minio
COPY --from=build /src/LICENSE /usr/share/licenses/minio/LICENSE
LABEL org.opencontainers.image.source="https://github.com/minio/minio" \
      org.opencontainers.image.revision="9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a" \
      org.opencontainers.image.licenses="AGPL-3.0-only"
ENTRYPOINT ["minio"]
