#!/bin/bash

# Build the Docker image
docker build -t mlops_image .

# Run the Docker container and mount the models directory
docker run -v $(pwd)/models:/app/models mlops_image
