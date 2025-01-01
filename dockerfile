FROM python:3.8

RUN mkdir /database
RUN mkdir /ingest

ADD config.json .
ADD rag_handler.py .
ADD requirements.txt .

RUN pip install -r requirements.txt

CMD [ "python", "./rag_handler.py", "--ollama-address", "http://192.168.1.100:11434", "--model", "llama3.2" ]