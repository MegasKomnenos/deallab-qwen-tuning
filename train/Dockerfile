# Use a specific CUDA runtime that matches the torch version
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script (the rewritten version from the previous step)
COPY train.py .

# Entrypoint ensures arguments are passed correctly
ENTRYPOINT ["python", "train.py"]