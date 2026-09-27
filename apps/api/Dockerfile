FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2, sentence-transformers, presidio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[eval]"

# Download spacy model for Presidio
RUN python -m spacy download en_core_web_lg 2>/dev/null || true

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
