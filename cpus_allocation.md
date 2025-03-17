## CPU-core allocations across containers

- For the configurations described in the paper, we set the following configurations on the 4 worker containers 
over a 48-core Intel Xeon E5-2670.
- Cumulatively, both ASP and BSP train over the 48 available cores.
- However, even for the same HL, the cpu-core sets allocated with ASP and BSP vary because the former trains with 
an additional 8-core parameter server while the latter runs in a decentralized manner.
- Parameter server in ASP training was allocated cpu-core sets ```0-7```.
- Before running/deploying a container, the ```--cpuset-cpus``` argument specifies the specific core sets to use.
  For e.g., ```docker run --name worker1 --cpuset-cpus 0-11 -d omnilearn-image:v1.0 /bin/bash``` spawns container
  ```worker1``` with 12 cores from 0-11.
- To trigger heterogeneity by varying $\textit{HL}s$ on already running containers, run docker update with the
  same argument. For e.g., ```docker update --cpuset-cpus 0-5 worker1``` updates the allocation of worker1 to use
  6 cores (0-5).

| HL  | Method | Worker #1 core-set | Worker #2 core-set | Worker #3 core-set | Worker #4 core-set |
|-----|--------|--------------------|--------------------|--------------------|--------------------|
| HL1 | BSP    | 0-11               | 12-23              | 24-35              | 36-47              |
| HL1 | ASP    | 8-17               | 18-27              | 28-37              | 38-47              |
| HL2 | BSP    | 0-11               | 12-23              | 24-31              | 32-47              |
| HL2 | ASP    | 8-15               | 16-23              | 24-31              | 32-47              |
| HL4 | BSP    | 0-8                | 9-16               | 17-22              | 23-47              |
| HL4 | ASP    | 8-17               | 18-27              | 28-31              | 32-47              |
| HL8 | BSP    | 0-5                | 6-11               | 12-15              | 16-47              |
| HL8 | ASP    | 8-11               | 12-15              | 16-19              | 20-47              |


## CPU configuration over 32 containers:

- For ResNet18/AlexNet experiments across 32 worker containers in BSP training. We spawn these containers across 4 servers, each
  eqipped with a 48-core CPU. To emulate $\textit{HL8}$, we allocate the workers as follows:

 | Worker ID  | Server ID | CPU-cores | CPU-set |
 |------------|-----------|-----------|---------|
 | Worker #1  | 1         | 2         | 0-1     |
 | Worker #2  | 1         | 2         | 2-3     |
 | Worker #3  | 1         | 2         | 4-5     |
 | Worker #4  | 1         | 2         | 6-7     |
 | Worker #5  | 1         | 6         | 8-13    |
 | Worker #6  | 1         | 6         | 14-19   |
 | Worker #7  | 1         | 6         | 20-25   |
 | Worker #8  | 1         | 6         | 26-31   |
 | Worker #9  | 1         | 2         | 32-33   |
 | Worker #10 | 1         | 2         | 34-35   |
 | Worker #11 | 1         | 2         | 36-37   |
 | Worker #12 | 2         | 6         | 0-5     |
 | Worker #13 | 2         | 6         | 6-11    |
 | Worker #14 | 2         | 6         | 12-17   |
 | Worker #15 | 2         | 6         | 18-23   |
 | Worker #16 | 2         | 16        | 24-39   |
 | Worker #17 | 3         | 32        | 0-31    |
 | Worker #18 | 3         | 2         | 32-33   |
 | Worker #19 | 3         | 2         | 34-35   |
 | Worker #20 | 3         | 2         | 36-37   |
 | Worker #21 | 1         | 4         | 38-41   |
 | Worker #22 | 3         | 4         | 38-41   |
 | Worker #23 | 4         | 4         | 0-3     |
 | Worker #24 | 4         | 4         | 4-7     |
 | Worker #25 | 4         | 6         | 8-13    |
 | Worker #26 | 4         | 6         | 14-19   |
 | Worker #27 | 2         | 4         | 40-43   |
 | Worker #28 | 4         | 4         | 20-23   |
 | Worker #29 | 4         | 4         | 24-27   |
 | Worker #30 | 4         | 4         | 28-31   |
 | Worker #31 | 4         | 4         | 32-35   |
 | Worker #32 | 4         | 4         | 36-39   |