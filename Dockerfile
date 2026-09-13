FROM python:3.11-slim

# Install system dependencies required for igraph and cairocffi
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Gorse
RUN wget -q https://github.com/gorse-io/gorse/releases/download/v0.5.11/gorse_linux_amd64.zip && \
    unzip gorse_linux_amd64.zip -d /tmp/gorse && \
    mv /tmp/gorse/gorse-in-one /usr/local/bin/ && \
    rm -rf /tmp/gorse gorse_linux_amd64.zip

WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry

# Prevent poetry from creating virtualenvs globally before we copy files
RUN poetry config virtualenvs.create false

# Copy the rest of the application
COPY . .

# Remove any host-generated lockfile to prevent Python version mismatch
RUN rm -f poetry.lock

# Lock and install dependencies natively inside the 3.11 container
RUN poetry lock && poetry install --no-interaction --no-ansi


RUN chmod +x /app/entrypoint.sh

# Use entrypoint to boot background services before main execution
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
