FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data ./data
ENV MADLANGA_DB=/srv/data/commission_research.db PORT=8000
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
