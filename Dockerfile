FROM python:3.11-slim

WORKDIR /app

# keep python output buffered and avoid writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps (minimal) and Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repo into image
COPY . .

# Default port for the container runtime (Fly sets PORT env)
ENV PORT=8080
EXPOSE 8080

# Start Streamlit on the container port and listen on all interfaces
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port ${PORT} --server.address 0.0.0.0"]
