# Build stage
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build the package
RUN pip install --user --no-cache-dir .

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code - all modules are now at root level
COPY --from=builder /build/api /app/api
COPY --from=builder /build/bio /app/bio
COPY --from=builder /build/bridge /app/bridge
COPY --from=builder /build/consciousness /app/consciousness
COPY --from=builder /build/core /app/core
COPY --from=builder /build/creativity /app/creativity
COPY --from=builder /build/emotion /app/emotion
COPY --from=builder /build/ethics /app/ethics
COPY --from=builder /build/identity /app/identity
COPY --from=builder /build/learning /app/learning
COPY --from=builder /build/memory /app/memory
COPY --from=builder /build/orchestration /app/orchestration
COPY --from=builder /build/quantum /app/quantum
COPY --from=builder /build/reasoning /app/reasoning
COPY --from=builder /build/symbolic /app/symbolic
COPY --from=builder /build/trace /app/trace
COPY --from=builder /build/voice /app/voice
COPY --from=builder /build/main.py /app/main.py
COPY --from=builder /build/scripts /app/scripts
COPY --from=builder /build/examples /app/examples

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV LUKHAS_ENV=production

# Create non-root user
RUN useradd -m -s /bin/bash lukhas && \
    chown -R lukhas:lukhas /app

USER lukhas

# Expose ports
EXPOSE 8000 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Set entrypoint
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command  
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000"]