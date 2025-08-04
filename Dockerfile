# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Create directories for model cache
RUN mkdir -p /app/artifacts /app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Pre-download base model to avoid download timeouts during runtime
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-cnn_dailymail'); \
    model = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-cnn_dailymail')"

# Run quick training to create a fine-tuned model (optional, uncomment if desired)
# RUN python main.py --stage all --quick-train

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app \
    && chown -R app:app /app/.cache
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
