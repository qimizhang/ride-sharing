# ride-sharing
Summer 2022 reasearch about ride sharing VRP

The two folders contain codes and experiment results of two directions of a ride-sharing problem.

The first direction is 

Formulate VRP with time constraints as IP +  Learning to cut, but VRP with time factors is a MIP problem which cannot be solved by Learning to cut.
So, we only solve a matching problem with this RL method.

The second direction is

Construction RL, Pointer Network + Policy Gradient, with time reward function. We made experiments in a 4-request and a 10-request scale problems.
