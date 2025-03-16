## CPU core allocations across worker containers

- For the configurations described in the paper, we set the following configurations on the 4 worker containers 
over a 48-core Intel Xeon E5-2670.
- Cumulatively, both ASP and BSP train over the 48 available cores.
- However, even for the same HL, the cpu-core sets allocated with ASP and BSP vary because the former trains with 
an additional 8-core parameter server while the latter runs in a decentralized manner.
- Parameter server in ASP training was allocated cpu-core sets ```0-7```.

  <div align="center">

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

  </div>