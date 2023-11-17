# FROM ubuntu:23.10
FROM python:3.9.17

# no need for conda or venv
WORKDIR /app

# copy the whole code directory
COPY . /app

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=api/app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]