# services/fastapi/Dockerfile  (or repo-root/Dockerfile if you prefer single service)
FROM python:3.11-slim

# Avoid Python buffering so logs flush immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build deps if needed (kept minimal)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . /app

# Create a non-root user and use it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Cloud Run will set PORT environment variable; provide a default for local dev
ENV PORT=8080
EXPOSE 8080

# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:$PORT", "main:app", "--timeout", "120"]
# Use shell form so env var expansion works
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
