FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TZ=Africa/Cairo
WORKDIR /app

COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY orvana/ orvana/
COPY scripts/ scripts/
COPY predict.py evaluate.py ./

# data/, incoming/, models/, state/ and outputs/ are mounted as volumes (see docker-compose.yml)
RUN useradd --create-home app && mkdir -p state outputs && chown -R app /app
USER app

CMD ["python", "-m", "orvana.scheduler"]
