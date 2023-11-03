# FROM ubuntu:23.10
FROM python:3.9.17

# no need for conda or venv
WORKDIR /app

# copy the whole code directory
COPY . /app

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Define the volume mount point
VOLUME /app/models

CMD ["pytest"]