FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV CHROMA_DB_DIR=/app/chroma_db
ENV EXEC_TIMEOUT_SECONDS=10

CMD ["python", "orchestration/coder_orchestrator.py", "Write a Python function add(a, b) and test it."]