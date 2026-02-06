FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY crawler.py /app/crawler.py
COPY verify_mongo.py /app/verify_mongo.py
COPY html_cleaner.py /app/html_cleaner.py
COPY tmp.py /app/tmp.py

CMD ["python", "/app/crawler.py", "/config/config.yaml"]
