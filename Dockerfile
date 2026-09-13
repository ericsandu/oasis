FROM python:3.11-slim

# Install system dependencies required for igraph and cairocffi
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the rest of the application
COPY . .

# Use pip to install the project and its dependencies (ignoring poetry.lock strictness)
RUN pip install --no-cache-dir .

# Install matplotlib for graphing outputs in scratch tests
RUN pip install --no-cache-dir matplotlib

# Default command
CMD ["bash"]
