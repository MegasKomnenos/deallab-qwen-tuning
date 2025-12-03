FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training logic
COPY train.py .

# Default entrypoint (can be overridden by KFP args)
ENTRYPOINT ["python", "train.py"]
