FROM python:3.8-bullseye

WORKDIR /app
COPY . /app

# Use apt-get instead of yum
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
