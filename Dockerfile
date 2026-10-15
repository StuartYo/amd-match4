# The final stage MUST start from the mandated base (checked by layer identity).
# Never squash this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    WA_LLM_BACKEND=transformers \
    WA_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
    WA_OUTPUT_DIR=/app/output \
    OMP_NUM_THREADS=4

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Pin the ROCm torch stack shipped in the base image so pip can never replace it
# with a CUDA build (the trap described in the Mini-Challenge 3 spec).
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm|numpy)==' > /tmp/torch-pins.txt \
 && cat /tmp/torch-pins.txt \
 && PIP_CONSTRAINT=/tmp/torch-pins.txt pip install -r /app/requirements.txt \
 && python3 -c "import torch, sys; print(torch.__version__); sys.exit(0 if torch.version.hip else 1)"

COPY app/ /app/
RUN mkdir -p /app/output /models && chmod +x /app/entrypoint.sh

# Weights are downloaded by the entrypoint at start-up (inside the 10 min budget)
# because the base image alone is close to the 60 GiB size limit.
CMD ["/app/entrypoint.sh"]
