# PYTHON MNIST DATASET BASED PROJECT
A small python project for learning and producing results from MNIST data set. Developing a Convolutional Neural Network From Scratch for MNIST Handwritten Digit Classification.

# Pre-requisites

## Docker must be Installed in your system in order to execute it

Get it from here: https://docs.docker.com/get-docker/

# Execution

**First of all pull the Docker image from Dockerhub**
```
docker pull syedsaadahmed2094/machine_learning_image:v1
```

**Second, Move to the directory where you have clone the GIT repository.**
```
cd /path to git repo/
```

**Thirdly create the container from that Image, It contains all the dependencies for the script to execute**
```
docker run -itd -v ${pwd}:/machine_learning_project [image_id] bash
```

**Get inside the container that is being created using the container ID**
```
docker attach [container ID]
```

**Finally you can find all the files of the repo inside the container, just go to**
```
cd /machine_learning_project
```
