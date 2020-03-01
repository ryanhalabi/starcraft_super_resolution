#!/bin/bash
yum -y install git
yum -y install python36

git clone https://github.com/ryanhalabi/starcraft_super_resolution

python3 -m pip install --upgrade pip
python3 -m pip install -r starcraft_super_resolution/requirements.txt
python3 -m pip install -e  starcraft_super_resolution/


export PATH=$PATH:/usr/local/bin/
python3 starcraft_super_resolution/run.py; tensorboard --logdir=/home/ec2-user/starcraft_super_resolution/upres --port=8080 --bind_all


