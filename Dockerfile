FROM python:3.10-slim

WORKDIR /app

# Install only the system packages we actually need
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# IMPORTANT: install dlib first (pre-built wheel)
RUN pip install --no-cache-dir dlib==19.24.2

# Then install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]

