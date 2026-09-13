FROM python:3.11-slim

# Install system dependencies required for igraph and cairocffi
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy the rest of the application
COPY . .

# Remove any host-generated lockfile to prevent Python version mismatch
RUN rm -f poetry.lock

# Lock and install dependencies natively inside the 3.11 container
RUN poetry lock && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Install matplotlib for graphing outputs in scratch tests
RUN pip install --no-cache-dir matplotlib

# Default command
CMD ["bash"]
