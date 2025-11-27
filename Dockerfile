FROM python:3.12-slim

WORKDIR /app

# Install dependencies for PDF parsing
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Ensure folders exist
RUN mkdir -p /app/chroma_db /app/Rules

EXPOSE 8000

CMD ["python", "main.py"]
