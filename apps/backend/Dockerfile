FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY apps/backend/requirements.txt /app/backend/requirements.txt

# Copy main package requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code
COPY apps/backend /app/backend
COPY alphagomoku /app/alphagomoku
COPY checkpoints/model.pt /app/model.pt

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "backend/run.py"]