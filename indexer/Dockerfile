FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x run.sh git-askpass.sh

ENV GIT_ASKPASS=/app/git-askpass.sh \
    GIT_TERMINAL_PROMPT=0 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["./run.sh"]