# ---------------------------
# 1. Use official Python image
# ---------------------------
FROM python:3.12-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 4. Copy requirements FIRST (layer caching)
# ---------------------------
COPY requirements.txt .

# ---------------------------
# 5. Install dependencies
# ---------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 6. Copy all your project files
# ---------------------------
COPY . .

# ---------------------------
# 7. Expose the port Azure uses
# ---------------------------
EXPOSE 8000

# ---------------------------
# 8. Start your Python bot server
# ---------------------------
CMD ["python", "main.py"]
