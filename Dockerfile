FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
ENV PYTHONPATH=/app/src \
    MODEL_DIR=/app/models/champion \
    DATA_PATH=/app/data/DemaReal_Sistema.parquet

# Model and data are mounted, not baked in: they are rebuilt with scripts/, not shipped in git
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "xmforecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
