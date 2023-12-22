FROM python:3.9-slim-buster

WORKDIR /app

COPY serving/requirements.txt requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "80"]