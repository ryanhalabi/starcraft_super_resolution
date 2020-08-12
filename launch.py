import boto3

from upres.utils.environment import env

"""
Script used to launch an EC2 instance, deploy the code, run
training, and host a Tensorboard dashboard.
"""

# instance_type = "t2.micro"
#instance_type = "g4dn.xlarge"
instance_type = "g4dn.4xlarge"
# instance_type = "p2.xlarge"

# deep learning amazon linux ami
ami_id = "ami-05e3c3618bcdf8a38"


# MODEL SETTINGS - Change these to configure your model.
# units "128,11 256,1 19"
# frames "64,9 128,1 17"

name = "color_units"
dataset = "units"
layers = "128,11 256,1 19"
scaling = 5
epochs = 20000000000
batch_size = 32
epochs_per_save = 2
overwrite = False

gpu_user_data = f"""#!/bin/bash
export PATH=$PATH:/home/ec2-user/anaconda3/bin
source activate tensorflow2_p36

git clone https://github.com/ryanhalabi/starcraft_super_resolution
git -C ./starcraft_super_resolution/ pull

python3 -m pip install --upgrade pip
python3 -m pip install -r starcraft_super_resolution/requirements.txt
python3 -m pip install -e starcraft_super_resolution/ --user

export AWS_ACCESS_KEY_ID={env.aws_access_key_id}
export AWS_SECRET_ACCESS_KEY={env.aws_secret_access_key}
export AWS_DEFAULT_REGION={env.aws_availability_zone}
export s3_bucket_name={env.aws_s3_bucket_name}

aws s3 sync {env.aws_s3_bucket_uri} /starcraft_super_resolution/upres/data

screen -S tensorboard -d -m bash -c "/home/ec2-user/anaconda3/envs/tensorflow2_p36/bin/tensorboard \
--logdir=/starcraft_super_resolution/upres/data/output --port=8080  --bind_all \
--max_reload_threads 1 --samples_per_plugin='images=0,scalars=0'"

screen -S training -d -m bash -c '\
export PATH=$PATH:/home/ec2-user/anaconda3/bin; \
source activate tensorflow2_p36; \
python3 /starcraft_super_resolution/run.py --name {name} --dataset {dataset} --layers {layers} \
--scaling {scaling} --epochs {epochs} --batch_size {batch_size} --epochs_per_save {epochs_per_save} --overwrite {overwrite}\
'
"""

print(gpu_user_data)

# Create and run EC2 instance
client = boto3.client("ec2", region_name=env.aws_availability_zone)
ec2 = boto3.resource("ec2", region_name=env.aws_availability_zone)
instance = ec2.create_instances(
    DryRun=False,
    ImageId=ami_id,
    MinCount=1,
    MaxCount=1,
    KeyName=env.aws_key_name,
    SecurityGroupIds=[env.aws_security_group_id],
    UserData=gpu_user_data,
    InstanceType=instance_type,
    SubnetId=env.aws_subnet_id,
)

instance_id = instance[0].id

response = client.describe_instances(InstanceIds=[instance_id])
public_dns = response["Reservations"][0]["Instances"][0]["PublicDnsName"]

while not public_dns:
    response = client.describe_instances(InstanceIds=[instance_id])
    public_dns = response["Reservations"][0]["Instances"][0]["PublicDnsName"]

print(f"""\nSSH COMMAND\nssh -i "{env.aws_key_name}.pem" ec2-user@{public_dns}""")
print(f"""\nTENSORBOARD\n{public_dns}:8080""")
print("\nGPU USAGE\nwatch -n 2 nvidia-smi")

input("Press enter to terminate instance.")
terminate_response = client.terminate_instances(InstanceIds=[instance_id])
