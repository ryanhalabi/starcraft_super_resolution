#!/bin/bash
yum -y install git
yum -y install python36

git clone https://github.com/ryanhalabi/starcraft_super_resolution

pip3 install --upgrade pip
pip3 install -r starcraft_super_resolution/requirements.txt
pip3 install -e  starcraft_super_resolution/


export PATH=$PATH:/usr/local/bin/
python3 starcraft_super_resolution/run.py; tensorboard --logdir=/home/ec2-user/starcraft_super_resolution/upres --port=8080 --bind_all


