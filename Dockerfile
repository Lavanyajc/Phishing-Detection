FROM python:3.8-bullseye

WORKDIR /app
COPY . /app

RUN apt update -y && apt install -y awscli

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
