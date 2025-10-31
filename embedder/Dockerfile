FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers torch
EXPOSE 7000
CMD ["python", "embedder_server.py"]
