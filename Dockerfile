FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Создаём файл api-keys.json, если его нет (сервер сгенерирует ключи при первом запуске)
# Можно также смонтировать свой файл через том или переменную окружения.
# RUN echo '{}' > api-keys.json

EXPOSE 6446

CMD ["sh", "-c", "python server.py --port ${PORT:-6446} --host 0.0.0.0"]
