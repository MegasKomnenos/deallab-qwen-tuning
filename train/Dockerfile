# UPDATED: Using PyTorch 2.3.0 to support modern Transformers/Qwen
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training logic
COPY train.py .

# Entrypoint
ENTRYPOINT ["python", "train.py"]