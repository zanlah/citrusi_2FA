
FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0


CMD ["flask", "run"]
