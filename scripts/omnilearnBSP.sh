#!/bin/bash

PARENT_DIR="$(dirname "$PWD")"
cd $PARENT_DIR

lr=0.15
gamma=0.2
momentum=0.7
weightdecay=1e-4
model='resnet18'
dataset='food101'
#lr=0.04
#gamma=0.1
#momentum=0.7
#weightdecay=1e-4
#model='vgg11'
#dataset='places365'

backend='gloo'
worldsize=4
initmethod='tcp'
trainfreq=200
testfreq=400
bsz=32
testbsz=32
globalB=$((bsz * worldsize))
user=$USER
hostname=$(uname -n)
sshstr=" $user@$hostname"

for i in $(seq 1 $worldsize)
do
  procrank=$(($i-1))
  container="worker"$(($i))
  ssh $sshstr "docker exec $container sh -c 'cd /OmniLearn && python3 -m omnilearn_pytorch.dynamic_controller.run_omnilearnBSP --determinism --asyncop --bsp --lr=$lr --gamma=$gamma --momentum=$momentum --weight-decay=$weightdecay --model=$model --dataset=$dataset --bsz=$bsz --backend=$backend --test-bsz=$testbsz --rank=$procrank --world-size=$worldsize --init-method=$initmethod --trainfreq=$trainfreq --testfreq=$testfreq --globalB=$globalB' &" &
  echo "deploying BSP job on container $container"
  sleep 1
done