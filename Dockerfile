# Base image
FROM python:3.12-slim

# Work directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY . .

# Ensure chroma directory exists
RUN mkdir -p /app/chroma_db
RUN mkdir -p /app/Rules

# Expose port
EXPOSE 8000

# Start bot
CMD ["python", "main.py"]
