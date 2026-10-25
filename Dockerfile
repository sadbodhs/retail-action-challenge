# Inference/training image for the RetailAction challenge entry.
# Public base (torch + CUDA), pinned by digest; everything else is pip-installed.
FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
RUN pip install --no-cache-dir av==19.0.2 kaggle==2.2.4
# timm for ImageNet-22k / ConvNeXt-V2 backbones; --no-deps keeps the base image's torch stack untouched
RUN pip install --no-cache-dir --no-deps timm==1.0.15
RUN ln -sf $(command -v python3) /usr/local/bin/python
WORKDIR /workspace
ENTRYPOINT []
