FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=6446
EXPOSE 6446

CMD ["python", "server.py", "--host", "0.0.0.0"]
