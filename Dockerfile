# FROM ubuntu:23.10
FROM python:3.9.17

# no need for conda or venv
WORKDIR /app

# copy the whole code directory
COPY . /app
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /app/requirements.txt
# need python

# Define the volume mount point
VOLUME /app/models

# requirements installation
CMD [""python", "plot_digits_classification.py""]