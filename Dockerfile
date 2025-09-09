FROM python:3.12.11-slim-bookworm

# Install system dependencies (libgl1 replaces libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 nano \
    && rm -rf /var/lib/apt/lists/*

    
# -------------------------
# Python Dependencies (pinned versions)
# -------------------------
COPY wheels /wheels
# RUN pip install --no-cache-dir /wheels/*.whl    

RUN pip install --no-cache-dir \
    python-doctr \
    paddleocr \
    paddlepaddle \
    opencv-python \
    pandas \
    tabulate \
    matplotlib \
    jupyter \
    notebook \
    fastapi \
    uvicorn

WORKDIR /app
COPY . /app

CMD ["python", "api.py"]
