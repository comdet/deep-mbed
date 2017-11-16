#!/bin/bash
#######################################################################
######## enable camera first by running 'sudo raspi-config' ###########
#######################################################################
sudo apt-get update
sudo apt-get install -y python-dev python-setuptools python-numpy python-pip python-opencv python-picamera
sudo apt-get install -y libopenblas-dev liblapack-dev libopencv-dev
###############################
## install pre-compile mxnet ##
###############################
cd ~
wget https://github.com/comdet/deep-mbed/raw/master/mxnet.tar.gz
tar -zxf mxnet.tar.gz
cd incubator-mxnet/python
sudo -H pip install --upgrade pip
sudo -H pip install -e .
printf "import mxnet\nprint mxnet.__version__" | python
cd ~
mkdir tensorflow
cd tensorflow
#benchmark test speed alexnet
python incubator-mxnet/example/image-classification/benchmark_score.py

########################
## install tensorflow ##
########################
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp27-none-linux_armv7l.whl
sudo pip install tensorflow-1.1.0-cp27-none-linux_armv7l.whl
wget https://raw.githubusercontent.com/samjabrahams/tensorflow-on-raspberry-pi/master/benchmarks/inceptionv3/classify_image_timed.py

#benchmark test speed inception-net-v3
cd ~
python deep-mbed/classify_image_timed.py --num_runs=10 --warmup_runs=3