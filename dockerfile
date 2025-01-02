FROM python:3.12

COPY . ./app
RUN mkdir /app/database
RUN mkdir /app/ingest
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "./app.py"]