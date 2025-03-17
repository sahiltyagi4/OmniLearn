#!/bin/bash

PARENT_DIR="$(dirname "$PWD")"
cd $PARENT_DIR

lr=0.1
gamma=0.1
momentum=0.9
weightdecay=1e-4
model='resnet50'
dataset='caltech256'
testbsz=128
# 1 parameter server and 4 workers
worldsize=5
trainfreq=200
testfreq=400
# list of comma separated values of initial worker batch-sizes across 4 containers
bszlist="32,32,32,32"
user=$USER
hostname=$(uname -n)
sshstr=" $user@$hostname"

for i in $(seq 1 $worldsize)
do
  procrank=$(($i-1))
  container="aspworker"$(($i))
  ssh $sshstr "docker exec $container sh -c 'cd /OmniLearn && python3 -m omnilearn_pytorch.dynamic_controller.run_omnilearnASP --determinism --asyncop --lr=$lr --gamma=$gamma --momentum=$momentum --weight-decay=$weightdecay --model=$model --dataset=$dataset --test-bsz=$testbsz --rank=$procrank --world-size=$worldsize --bszlist=$bszlist --trainfreq=$trainfreq --testfreq=$testfreq' &" &
  echo "deploying ASP job on container $container"
  sleep 1
done