# Production-optimized Dockerfile for Luceron AI Manager Agent
FROM python:3.13-slim

# Build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Labels for container metadata
LABEL org.opencontainers.image.title="Luceron AI Manager Agent"
LABEL org.opencontainers.image.description="Central orchestration layer for the Luceron AI eDiscovery Platform"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Luceron AI"

# Environment optimization
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PYTHONPATH=/app
ENV PORT=8081

# Security and performance settings
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

WORKDIR /app

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies
    gcc \
    g++ \
    # Runtime dependencies
    curl \
    ca-certificates \
    # Security updates
    && apt-get upgrade -y \
    # Clean up to reduce image size
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user early for security
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent

# Copy and install Python dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Clean pip cache to reduce image size
    && pip cache purge

# Copy application files
COPY main.py .
COPY prompts/ prompts/

# Set ownership and switch to non-root user
RUN chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 8081

# Enhanced health check with custom endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8081/status || exit 1

# Security: Run with minimal privileges
USER agent

# Optimized startup command
CMD ["python", "-u", "main.py"]