FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
