# PYTHON MNIST DATASET BASED PROJECT
A small python project for learning and producing results from MNIST data set


# Pre-requisites

## Docker must be Installed in your system in order to execute it

Get it from here: https://docs.docker.com/get-docker/

## Execution

**First of all pull the Docker image from Dockerhub**
```
docker pull
```

**Second, create the container from that Image, It contains all the dependencies for the script to execute**
```
docker run -itd -v ${pwd}:/machine_learning_project [image_id] bash
```
