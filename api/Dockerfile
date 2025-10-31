FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn httpx
EXPOSE 8080
CMD ["python", "rag_api.py"]
