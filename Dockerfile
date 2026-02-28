# Scanner ULTRA v5.0.0 — Production Docker Image
# Multi-stage build for minimal image size

# --- Builder stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# --- Runtime stage ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# System deps for OpenCV, MediaPipe (Debian 12+: libgl1-mesa-glx → libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ src/
COPY weights/ weights/

ENV SCANNER_ENV=production
ENV SCANNER_HOST=0.0.0.0
ENV SCANNER_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "scanner.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
