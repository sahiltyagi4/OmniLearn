<div align="center">

# $\texttt{OmniLearn}$ : A Framework for Distributed Deep Learning over Heterogeneous Clusters

Implementation of $\texttt{OmniLearn}$, published at _IEEE Transactions on Parallel and Distributed Systems (TPDS)_, 2025.

<img src="figures/omnilearn_design.jpg" alt="Description" style="width: 600px; height: auto;">

</div>

$\texttt{OmniLearn}$ overview: Initially, fast and slow workers (p, q) train on the same mini-batch $b_{p} = b_{q}$ (breadth of 
‘Compute’ block) that leads to stragglers (BSP) or staleness (ASP) as compute times $t_{p} < t_{q}$ (length of ‘Compute’). 
Controller adjusts mini-batches to equalize compute times, i.e., $t_{p}^{'} \approx t_{p}^{'}$ : $b_{p}^{'} > b_{q}^{'}$ 
since p > q from a computational standpoint.

## Docker installation and container deployment
- To emulate compute nodes with varying heterogeneity, install docker as follows:
  ```
  git clone https://github.com/sahiltyagi4/OmniLearn
  cd OmniLearn
  chmod a+x docker_install.sh
  ./docker_install.sh
  ```

- Build a docker image with installed dependencies, spawn containers from ```Dockerfile```.
  ```
  cd OmniLearn/
  // create docker image named omnilearn-image version v1.0
  docker build -t omnilearn-image:v1.0 .
  // check if docker image is built
  docker images
  // create containers. The following spawns 4 containers with HL1 over a 48-core CPU.
  docker run -i --name worker1 --cpuset-cpus 0-11 --memory 12g --shm-size 1g -d omnilearn-image:v1.0 /bin/bash
  docker run -i --name worker2 --cpuset-cpus 12-23 --memory 12g --shm-size 1g -d omnilearn-image:v1.0 /bin/bash
  docker run -i --name worker3 --cpuset-cpus 24-35 --memory 12g --shm-size 1g -d omnilearn-image:v1.0 /bin/bash
  docker run -i --name worker4 --cpuset-cpus 36-47 --memory 12g --shm-size 1g -d omnilearn-image:v1.0 /bin/bash
  ```

- Create a docker swarm network to enable communication between nodes:
  ```
  // run this on the parameter server node or any one of the workers
  docker swarm init --advertise-addr **server-ip**
  // run docker swarm join command returned by the above on the remaining nodes
  // create docker swarm network
  docker network create --driver overlay --attachable omninet
  ```

## Job deployment

### Emulating heterogeneity across containers
- Heterogeneity is introduced by varying the cpu-cores allocated to a container.
- Dynamic heterogeneity simulations at various HLs in ResNet18, ResNet50, AlexNet, VGG11 and GPT-2 is implemented in ```pytorch.helper.dynamicbatching.DynamicHeterogeneityEmulator``` class.
  
   | HL             | Method      | <center>Container #1 CPU-cores | <center>Container #2 CPU-cores | <center>Container #3 CPU-cores | <center>Container #4 CPU-cores  |
   |----------------|-------------|--------------------------------|--------------------------------|--------------------------------|---------------------------------|
   | $\textit{HL1}$ | <center>BSP | <center>12                     | <center>12                     | <center>12                     | <center>12                      |
   | $\textit{HL1}$ | <center>ASP | <center>10                     | <center>10                     | <center>10                     | <center>10                      |
   | $\textit{HL2}$ | <center>BSP | <center>12                     | <center>12                     | <center>8                      | <center>16                      |
   | $\textit{HL2}$ | <center>ASP | <center>8                      | <center>8                      | <center>8                      | <center>16                      |
   | $\textit{HL4}$ | <center>BSP | <center>9                      | <center>9                      | <center>6                      | <center>24                      |
   | $\textit{HL4}$ | <center>ASP | <center>10                     | <center>10                     | <center>4                      | <center>16                      |
   | $\textit{HL8}$ | <center>BSP | <center>6                      | <center>6                      | <center>4                      | <center>32                      |
   | $\textit{HL8}$ | <center>ASP | <center>4                      | <center>4                      | <center>4                      | <center>28                      |

- ```DynamicHeterogeneityEmulator``` works by allocating cpu-sets to 4 worker containers, as per the above table. Users can change the number of nodes or degree of heterogeneity with ```container_cpuconf``` dictionary.
- To emulate dynamic heterogeneity across the 4 containers, execute ```dynamicbatching.py``` on the host server running the containers:
  ```
  cd OmniLearn/scripts
  chmod a+x exec_HL.sh
  ./exec_HL.sh
  ```
  
  <div align="center">
    <img src="figures/dynHLroutineres18vgg.jpg" alt="Description" style="width: 420px; height: 275px;">
    <img src="figures/dynHLroutineres50alexnet.jpg" alt="Description" style="width: 420px; height: 275px;">
  </div>
  
- The emulator triggers HLs for the above models (as well as for GPT-2 following the same trajectory as ResNet18).

### Launching training
  #### BSP Training:

  #### ASP Training:


## Citation
Please refer to the following to understand the current chosen configurations, implementation and design:
````
@article{TyagiOmniLearn2025,
         title={OmniLearn: A Framework for Distributed Deep Learning over Heterogeneous Clusters},
         author={Sahil Tyagi and Prateek Sharma},
         journal={IEEE Transactions on Parallel and Distributed Systems (TPDS)},
         year={2025}}
````