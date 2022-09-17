# Learning to cut Project


## Installation
The only dependency is gurobipy. Install gurobipy:

```
$ conda install -c gurobi gurobi
```

In addition, you need an academic license from gurobi. After getting the license, go to the license page.

(https://www.gurobi.com/downloads/end-user-license-agreement-academic/)

 
## WandB for Visualizaition
Class labs have made extensive use of wandb to familiarize you with some great machine learning visualization tools. You are encouraged to use wandb in the development of this project. We will soon provide further information about leaderboard for the project.


## Training Performance Evaluation
There are two environment settings on which your training performance will be evaluated. These can be loaded by using the following two configs (see example.py). Each mode is characterized by a set of parameters that define the cutting plane environment.


## Generalization
For the first phase of the project, your task is to reach the best possible performance on the two training modes described above. We will introduce another test mode for the environment later in the semester where your agent will be tested on a cutting plane environment with unseen instances (of size 60 by 60).

## Generating New Instances

To make sure your algorithm generalizes to instances beyond those in the instances folder, you can create new environments with random IP instances and train/test on those. To generate new instances, run the following script. This will create 100 new instances with 60 constraints and 60 variables.
In our VRP, num-V can be the number of dispatching variables like the vehicles. Num-c is the number of constraints, we need to write the IP formulation of 
ride sharing matching problem.

```
$ python generate_randomip.py --num-v 60 --num-c 60 --num-instances 100
```

The above instances will be saved in a directory named 'instances/randomip_n60_m60'. Then, we can load instances into gym env and train a cutting agent. The following code loads the 50th instance and run an episode with horizon 50:

```
python testgymenv.py --timelimit 50 --instance-idx 50 --instance-name randomip_n60_m60
```

We should see the printing of step information till the episode ends.

If you do not provide --instance-idx, then the environment will load random instance out of the 100 instances in every episode. It is sometimes easier to train on a single instance to start with, instead of a pool of instances.


To train and test the model: 
``` 
python rl_cuts.py 
```
