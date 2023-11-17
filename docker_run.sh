#!/bin/bash

# Build the Docker image
docker build -t mlops_image .

# Run the Docker container and mount the models directory
docker run -p 5000:5000 mlops_image
