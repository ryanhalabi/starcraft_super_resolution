# bash scripts to run to launch model training and tensorboard 

sudo yum -y install git
sudo yum -y install python36


git clone https://github.com/ryanhalabi/starcraft_super_resolution

sudo python3 -m pip install --upgrade pip

sudo python3 -m pip install -r starcraft_super_resolution/requirements.txt

sudo python3 -m pip install -e  starcraft_super_resolution/

python3 starcraft_super_resolution/run.py

tensorboard --logdir=/home/ec2-user/starcraft_super_resolution/upres --port=8080 --bind_all
