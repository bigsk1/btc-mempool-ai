# Use an official Python runtime as the base image
FROM python:3.12-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file initially
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Start a new stage for the final image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install network tools in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy only necessary files
COPY app.py .

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Add a health check
HEALTHCHECK CMD curl -f https://mempool.space/api/v1/prices || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]