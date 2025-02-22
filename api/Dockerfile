# Use the official PyTorch image with CUDA support, which includes Python
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip libsndfile1 build-essential cmake libgtk-3-dev \
    cargo libboost-all-dev git gcc g++ python3-dev && \
    rm -rf /var/lib/apt/lists/*


# Create and switch to a working directory
WORKDIR /app
ENV PYTHONPATH=/app
# Copy requirements.txt from the build context into the container
COPY ./requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the entire context into /app
COPY . /app
# Debug: List files to confirm structure
RUN find /app -type f
# Expose the port
EXPOSE 8009

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]