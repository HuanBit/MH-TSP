# TSP using scikit-OPT

This document is an assignment for the Intelligent Optimization Methods course.

The assignment introduces the optimization effect of five meta-heuristic algorithms, genetic algorithm, immune algorithm, simulated annealing algorithm, ant colony algorithm and particle swarm optimization algorithm for TSP. 

This code repository is the code used in the experiment part of the assignment.



## Setup environments

```code 
conda create -n MH_TSP python==3.7.5
conda activate MH_TSP
pip install scikit-OPT
```



## file structure

```
MH-TSP
+--integrate_MH_TSP.py   // Test the optimization performance of the algorithm
+--readme.md
+--results
| +--figs
| | +--3map_{METHOD}_{NUM_POINTS}_{MAX_ITERATION}.png	// cmp2
| | +--5method_{NUM_POINTS}_{MAX_ITERATION}_cmp.png     // cmp1
| | +--{METHOD}_{NUM_POINTS}_{MAX_ITERATION}.png        // cmp3
| +--points_coordinate_{NUM_POINTS}{SET}.npy
| +--distance_matrix_{NUM_POINTS}{SET}.npy
+--utils
| +--compare.py			  // Carry out comparative analysis and plot(different map/set)
| +--generate.py          // Generate TSP instance
| +--print_tree.py        // Generate a tree file structure
```



