FROM python:3.9-slim-buster

WORKDIR /app

COPY training/requirements.txt requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

CMD ["python", "training/main.py"]