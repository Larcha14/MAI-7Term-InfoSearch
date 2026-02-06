FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plot_zipf.py .

CMD ["python", "/app/plot_zipf.py"]