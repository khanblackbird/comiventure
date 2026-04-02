FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

WORKDIR /app

# Install PyTorch with CUDA (large layer, cached separately)
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install AI model dependencies (separate layer for caching)
RUN python -m pip install --no-cache-dir diffusers transformers accelerate safetensors Pillow python-multipart "httpx[socks]"

# Install app dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/data/models
ENV TRANSFORMERS_CACHE=/app/data/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
