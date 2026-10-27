FROM python:3.13-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install the package and the server dependencies
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

# the API code and the model weights
COPY app ./app

EXPOSE 8000

# Docker calls /health to check that the service is running
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
