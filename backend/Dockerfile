FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for geospatial libraries and Postgres
RUN apt-get update && apt-get install -y libpq-dev gcc netcat-traditional && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
