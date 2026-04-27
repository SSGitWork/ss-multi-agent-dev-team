FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY orchestration/ ./orchestration/
COPY docs/ ./docs/

# Create runtime directories
RUN mkdir -p /app/workspace /app/chroma_store /app/docs/cost_reports

# Health check script
RUN echo '#!/bin/bash\npython -c "import agents.config; print(\"healthy\")"' > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["bash", "/app/healthcheck.sh"]

ENTRYPOINT ["python", "-m", "orchestration.runner"]
CMD ["--interactive"]