# 1. Base image: official PyTorch image with CUDA and cuDNN
# This is required to run TensorRT and onnxruntime-gpu
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies
# These packages are required because opencv-python needs graphics libraries on Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (to benefit from Docker cache and speed up later builds)
COPY Deployment/requirements.txt .

# 5. Install Python packages
# Use --no-cache-dir to keep the final image smaller
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the project folders into the container
COPY Src/ ./Src/
COPY Weights/ ./Weights/
COPY Deployment/ ./Deployment/

# 7. Set environment variables
# This lets Python import from Src (from Src.models import ...)
ENV PYTHONPATH=/app

# 8. Expose the FastAPI port
EXPOSE 8000

# 9. Default command when the container starts
CMD ["uvicorn", "Deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
