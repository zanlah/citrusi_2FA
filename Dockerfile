
FROM python:3.9-slim


WORKDIR /app


COPY . /app


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=127.0.0.1


CMD ["flask", "run"]
