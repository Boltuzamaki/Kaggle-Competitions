# FREUID Challenge 2026 -- reproducible inference image.
# Build (needs network, for pip installs and to pull the base image):
#   docker build -t freuid-repro:local .
# Run (no network, matches the competition's sandbox contract):
#   docker run --rm --network none \
#     -v /path/to/flat/test/images:/data:ro \
#     -v "$(pwd)/out:/submissions" \
#     freuid-repro:local
# (add --gpus all before --rm if a CUDA GPU is available; falls back to CPU otherwise)

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY weights ./weights

ENV FREUID_DATA_DIR=/data
ENV FREUID_OUT_DIR=/submissions

ENTRYPOINT ["python", "inference.py"]
