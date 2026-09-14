FROM python:3.11-slim

WORKDIR /app/backend

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py /app/backend/app.py
COPY backend/search.py /app/backend/search.py
COPY backend/image_processor.py /app/backend/image_processor.py

COPY index /app/index
COPY deploy_images /app/val2017/val2017

CMD uvicorn app:app --host 0.0.0.0 --port $PORT