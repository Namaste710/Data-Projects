FROM docker.io/library/python:3.7-slim-buster

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./src .

CMD ["python", "training.py"]
