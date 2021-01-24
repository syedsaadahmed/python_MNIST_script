FROM tensorflow/tensorflow:latest

MAINTAINER "Syed Saad Ahmed"

RUN pip uninstall tensorflow -y && pip install tensorflow==1.5

RUN pip install matplotlib && pip install sklearn && pip install keras==keras==2.1.5


apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update

apt install python3-pip

pip3 install --upgrade pip

pip3 install numpy scikit-learn matplotlib

pip3 install tensorflow

pip3 install keras

pip3 install --upgrade tensorflow==2.2



