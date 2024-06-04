
FROM python:3.9-slim

#RUN apt-get update && apt-get install -y libgl1-mesa-glx
#RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Setup SSH server
RUN mkdir /var/run/sshd
RUN useradd -m -s /bin/bash newuser && \
    echo 'newuser:ErikPustoslemsek' | chpasswd && \
    mkdir -p /home/newuser/.ssh && \
    chown -R newuser:newuser /home/newuser/.ssh

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config


WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000 22

ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0


CMD service ssh start && flask run
#CMD ["flask", "run"]
