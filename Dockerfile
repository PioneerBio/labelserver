FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libspatialindex-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY labelserver/ /app/labelserver/

EXPOSE 8889

CMD ["uvicorn", "labelserver.main:app", "--host", "0.0.0.0", "--port", "8889", "--workers", "4"]
