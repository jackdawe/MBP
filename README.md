# joliRL

This project aims to reproduce Mikael Henaff, Will Whitney and Yann LeCun's work on model-based planning.

In their research, they tested their algorithms on two tasks:

1. **GridWorld**, a state of the art toy example where the agent has to find its way in a grid by chosing whether to go up, down, left or right without colliding with an obstacle.

2. **Starship**, a more complex environment with an infinite state and action space. Many state of the art RL algorithms such as Q-Learning or TD-Lambda cannot be applied without using tricks such as discretizing the state space. The agent controls the thrusters of a small ship and has to fight the gravitational pull of the planets to reach the closest waypoint. Once the agent reaches the waypoint he has to fire the correct signal to indicate that he has reached this precise waypoint to collect a reward.  

For this work, I reproduced the environments as described in the paper. 

You can find it [here](https://arxiv.org/abs/1705.07177).

# General Information 

My code provides a certain amount of commands to easily use the main functionalities of this project. The most useful commands are documented down below. 

You will find [here](https://github.com/jackdawe/joliRL/tree/master/tests/baseline) a notebook with a full example on how to use the main commands on the Starship problem, from generating maps to running and testing the planner. 

You will also find 4 [here](https://github.com/jackdawe/joliRL/tree/master/Videos)

1. joliRL/project/ is the root directory for any path you have to provide as parameter.
2. To use one of my commands, first go to the root directory. Then follow the general architecture:
```
./project -cmd=COMMAND_NAME -myflag1=val1 -myflag2=val2 (...) -myflag100=val100 
```
3. Flags to indicate the path to a directory should end with a "/". 

# Installing Dependancies

You'll need to have the three libraries installed to run this code:

- Qt5
- libtorch
- gflags

## Qt5

I use Qt5 for the GridWorld and Starship GUIs. You can do so with the following command:
```
sudo apt-get install qt-sdk
```

## libtorch

libtorch is the PyTorch C++ API. You can install libtorch at [https://pytorch.org/](https://pytorch.org/) by selecting Linux, Conda, C++/Java and Cuda 10.1 and clicking on the second link. You should then unzip the directory and move it such as it is in the same directory as joliRL.

### gflags

You can see how to install gflags [here](https://gflags.github.io/gflags/). 

# GRIDWORLD TASK

## Map Generation

To generate a GridWorld map file that you can use for your tasks. 

Map files contain a simple matrix:
- 0 for an empty space
- 1 for an obstacle
- 2 for the goal

### Command name

gwmgen

### Parameters

- int **size**  : the size of the map. I advise you to stick to 8 and 16 as other sizes weren't tested. Default: 8 
- int **maxObst** : the maximum number of obstacles. The map will be generated with a random number of obstacles between 0 and maxObst. Please note that walls are automatically generated and are not taken into account in the maxObst count. Default: 1 
- string **map** : the path to the file that will be created. Default: root directory. 

### Example

To generate a map named **myMap** of size **16** with a maximum of **50** obstacles in the **GridWorld/Maps/** directory, execute the following command:
```
./project -cmd=gwmgen -size=16 -maxObst=50 -map=../GridWorld/Maps/myMap 
```
## Map pool generation

For my algorithms to generalize well, I needed to train my agent on several maps. You can generate a pool of training and test maps using this command. 

### Command name

gwmpgen

### Parameters

- int **size** : the size of the map. I advise you to stick to 8 and 16 as other sizes weren't tested. Default: 8
- int **maxObst** : the maximum number of obstacles. The map will be generated with a random number of obstacles between 0 and maxObst. Please note that walls are automatically generated and are not taken into account in the maxObst count. Default: 1
- string **mp** : the path to the directory that will contain the test and train map pools. A new directory is created if it does not already exists. Default: root directory.
- int **nmaps**: the number of maps to be generated (both train and test map pools will contain nmaps maps). Default: 1

### Example 

To create a directory named **MyMapPool** in the **GridWorld/Maps/** directory containing two sub directories (test and train) each containing **1000** maps of size **8** with a maximum of **10** obstacles:
```
./project -cmd=gwmpgen -size=8 -maxObst=10 -mp=../GridWorld/Maps/MyMapPool/ -nmaps=1000
```
## Showing the map using the implemented GUI

You can use the GUI to display the map on your screen.

### Command

**gwmshow**

### Parameters

- string **map** : the path to the file containing the map to be displayed. Default: root directory.

### Example

You can show the map named **map0** in the **GridWorld/Maps/MyMapPool/test/** directory using the following command line:
```
./project -cmd=gwmshow -map=../GridWorld/Maps/MyMapPool/test/map0
```
Here is the type of window that you should get:

![Map example](https://github.com/jackdawe/joliRL/blob/master/img/Screenshot%20from%202020-01-16%2015-03-14.png "This is how the map should look like with Qt!")

## Generating the data set

An agent appears at a random location on a map randomly chosen from a map pool and wanders randomly until he reaches a terminal state. Additionnal states beyond this terminal state are padded until reaching **T**. Actions, State t, State t+1 and rewards are recorded and are each stored in a tensor.

Every sample contains **T** transitions. This is useful when you want your agent to learn only from the initial state, using its own predictions as inputs afterwards.

A train and test set are generated using the train and test maps respectively.

Executing this command creates 8 files in the map pool directory:
- actionInputsTr.pt containing a { **n** x **trp** | **T** | 4 } tensor
- stateInputsTr.pt containing a { **n** x **trp** | **T** | **size** x **size** + 4 } tensor
- stateLabelsTr.pt containing a { **n** x **trp** | **T** | **size x **size** + 4 } tensor
- rewardLabelsTr.pt containing a { **n** x **trp** | **T** } tensor
- actionInputsTe.pt containing a { **n** x (1-**trp**) | **T** | 4 } tensor
- stateInputsTe.pt containing a { **n** x (1-**trp**) | **T** | **size** x **size** + 4 } tensor
- stateLabelsTe.pt containing a { **n** x (1-**trp**) | **T** | **size** x **size** + 4 } tensor
- rewardLabelsTe.pt containing a { **n** x (1-**trp**) | **T** } tensor

### Command

gwdsgen

### Parameters

- string **mp** : the path to the directory containing your train and test map pools. Default: root directory.
- int **nmaps** : the number of maps in the map pool. A value higher than the actual number of maps crashes the program. Default: 1
- int **n** : the size of the dataset. Overall, the dataset thus contrains T*n transitions. Default : 10000 
- int **T** : the number of transitions in a sample. If the agent reached a terminal state before the end of the T transitions, this same state is added to the State t and State t+1 tensors and 0 to the reward tensor. Default: 1
- float **wp** : Reaching the goal is a rare event, and gets rarer as the size of the map increases. For the model to see this happen more often, the agent is forced to make a transition towards the goal during wp*n episodes. Thus, wp should range from 0 to 1. Default: 0.1
- float **trp** : The training set's share of the dataset (ranging from 0 to 1). The testing set's share of the dataset is 1-trp. Default: 0.9

### Example

The following command uses the **1000** first maps of the **../GridWorld/Maps/MyMapPool/** map pool directory to generate 8 tensor files forming both the test and training set. The tensor' first two dimension are (trp x n | T) for the training set and ((1-trp) x n | T) for the test set. 10% of the dataset contains samples where the agent appears next to the goal and makes a transition towards it. 
```
./project -cmd=gwdsgen -mp=../GridWorld/Maps/MyMapPool/ -nmaps=1000 -n=10000 -T=10 -wp=0.1 -trp=0.9 
```
## Training a forward model

A neural network learns the physics of the GridWorld using a pre-generated dataset using the gwdsgen command. The model uses RGB images of the state as input, thus it automatically converts the state tensor dataset into a RGB image. 

### Command

gwmbfm

### Parameters

- string **mp** : the path to the directory contraining your 8 dataset tensors. These tensors must keep their original names. Default: root directory.
- int **sc1** : manages the number of feature maps of the convolutionnal layers. Default: 16
- float **lr** : learning rate. Default: 0.001
- int **n** : Number of iterations. Default: 10000
- int **bs** : batch size. Default: 32
- float **beta** : state loss multiplicative coefficient. Total loss = beta * stateloss + reward loss. Default: 1. 
- bool **asp** : If set to false, the neural network will only be provided with the initial state for each sample. The next states are then calculated using the model's previous predictions. Default: true.

## Gradient Based Planner

Command: ???
Parameters:
- K : The number of rollouts.
- T : The amount of time steps to be planned
- gs : The number of gradient steps to be performed
- lr : Learning rate

# STARSHIP COMMANDS

## Map Generation

To generate a map of the Starship environment. 

### Command name

ssmgen

### Parameters

- int **nplan**  : The number of planets. Expect an infinite loop if you fill up the available space. Default: 1 
- int **pmin** : The planet minimum radius in pixels. Default: 50
- int **pmax** : The planet maximum radius in pixels. Default: 100
- int **nwp**  : The number of waypoints. Should not be higher that 6 for GUI. Default: 3 
- int **rwp** : The radius of each waypoint in pixels. Default: 15
- string **map** : the path to the file that will be created. Default: root directory. 

### Example

To generate a map named **myMap** with **2** planets of radius ranging from **60** to **80**, **4** waypoints of radius **15** in the **Starship/Maps/** directory, execute the following command:
```
./project -cmd=ssmgen -nplan=2 -pmin=60 -pmax=80 -nwp=4 -rwp=15 -map=../Starship/Maps/myMap 
```
## Map Pool Generation

To generate a map pool of test and train maps of the Starship environment. 

### Command name

ssmpgen

### Parameters

- int **nplan**  : The number of planets. Expect an infinite loop if you fill up the available space. Default: 1 
- int **pmin** : The planet minimum radius in pixels. Default: 50
- int **pmax** : The planet maximum radius in pixels. Default: 100
- int **nwp**  : The number of waypoints. Should not be higher that 6 for GUI. Default: 3 
- int **rwp** : The radius of each waypoint in pixels. Default: 15
- string **mp** : the path to the directory that will contain the test and train map pools. A new directory is created if it does not already exists. Default: root directory.
- int **nmaps** : the number of maps to be generated (both train and test map pools will contain nmaps maps). Default: 1

### Example

To create a directory named **MyMapPool** in the **Starship/Maps/** directory containing two sub directories (test and train) each containing **100** maps using the **default values** for the planet and waypoint parameters:
```
./project -cmd=ssmpgen -mp=../Starship/Maps/myMapPool/ -nmaps=100 
```
## Showing the map using the implemented GUI

You can use the GUI to display the map on your screen.

### Command

**ssmshow**

### Parameters

- string **map** : the path to the file containing the map to be displayed. Default: root directory.

### Example

You can show the map named **MyMap** in the **Starship/Maps/MyMapPool/test/map0** using the following command line:
```
./project -cmd=gwmshow -map=../Starship/Maps/MyMapPool/test/map0
```
Here is the type of window that you should get:

![Map example](https://github.com/jackdawe/joliRL/blob/master/img/Screenshot%20from%202020-01-16%2016-37-06.png "This is how the map should look like with Qt!")

The blue circle represents the planet. The other colored circles represent the waypoints.

## Generating the data set

An agent appears at a random location on a map randomly chosen from a map pool and randomly fires his thrusters and lights his signal for 80 timesteps. Actions, State t, State t+1 and rewards are recorded and are each stored in a tensor.

Every sample contains **T** transitions. This is useful when you want your agent to learn only from the initial state, using its own predictions as inputs afterwards. 

A train and test set are generated using the train and test maps respectively.

Action tensors are generated with discrete actions being encoded as one-hot vectors and thrust vector values are put between 0 and 1 as the planner will only take actions between these bounds. You can chose from 4 distributions for your actions:

#### Uniform over thrust vector coordinates (dist = 0)

![dist1](https://github.com/jackdawe/joliRL/blob/master/img/dist1.png)

#### Uniform over thrust norm and angle (dist = 1)

![dist2](https://github.com/jackdawe/joliRL/blob/master/img/dist2.png)

#### Per trajectory isotopic gaussian over thrust vector coordinates (dist = 2)

On this plot you see the action distribution over one trajectory with a 0.1 standard deviation

![dist3](https://github.com/jackdawe/joliRL/blob/master/img/dist3.png)

#### Per trajectory isotopic gaussian over thrust norm and angle (dist = 3)

On this plot you also see the action distribution over one trajectory with a 0.1 standard deviation

![dist4](https://github.com/jackdawe/joliRL/blob/master/img/dist4.png)

#### Default

If **dist** is set to any other value, the thrusters will be turned off during dataset generation.  

Dataset generation leads to the creation of 8 files in the map pool directory:
- actionInputsTr.pt containing a { **n** x **trp** x 80/**T** | **T** | a } tensor
- stateInputsTr.pt containing a { **n** x **trp** x 80/**T** | **T** | s } tensor
- stateLabelsTr.pt containing a { **n** x **trp** x 80/**T** | **T** | 4 } tensor
- rewardLabelsTr.pt containing a { **n** x **trp** x 80/**T** | **T** } tensor
- actionInputsTe.pt containing a { **n** x (1-**trp**) x 80/**T** | **T** | a } tensor
- stateInputsTe.pt containing a { **n** x (1-**trp**) x 80/**T** | **T** | s } tensor
- stateLabelsTe.pt containing a { **n** x (1-**trp**) x 80/**T** | **T** | 4 } tensor
- rewardLabelsTe.pt containing a { **n** x (1-**trp**) x 80/**T** | **T** } tensor

s is the size of the state vector which is equal to 4 + 3 x (nPlanets + nWaypoints).

a is the size of the action vector which is equel to nWaypoints + 3. 

### Command

ssdsgen

### Parameters

- string **mp** : the path to the directory containing your train and test map pools. Default: root directory.
- int **nmaps** : the number of maps in the map pool. A value higher than the actual number of maps crashes the program. Default: 1
- int **n** : the number of trajectories. Overall, the dataset contrains **T** x **n** transitions. Default : 10000 
- int **T** : the number of transitions in a sample. **T** must divide 80 or the command won't execute itself. Default: 1
- float **wp** : Reaching a waypoint is a rare event. For the model to see this happen more often,  **wp** x **n** trajectories contain at least one transition towards a waypoint. **wp** should range from 0 to 1. Default: 0.1
- float **trp** : The training set's share of the dataset (ranging from 0 to 1). The testing set's share of the dataset is 1-**trp**. Default: 0.9
- int **dist** : The action distribution. Default: 0. 
- float **alpha** : How much you want to spread the actions beyond their original boundaries. Default: 1 
- float **sd** : The standard deviation of the gaussian distribution if **dist** is set to 2 or 3. Default: 0.25
- bool **woda** : you can disable the signal by setting this parameter to true. If so, the agent will receive a positive reward as long as he is on the waypoint. Default: false

### Example

The following command uses the **1000** first maps of the **../Starship/Maps/MyMapPool/** map pool directory to generate 8 tensor files forming both the test and training set. The tensor' first two dimensions are (**trp** x **n** x 80/**T** | **T**) for the training set and ((1-**trp**) x **n** x 80/**T** | **T**) for the test set. **10%** of the dataset contains samples where the agent passes on a waypoint. The actions follow a **per trajectory isotopic gaussian over thrust vector coordinates** with a standard deviation of **0.1**.
```
./project -cmd=ssdsgen -mp=../Starship/Maps/MyMapPool/ -nmaps=1000 -n=10000 -T=10 -wp=0.1 -trp=0.9 -dist=2 -sd=0.1 
```

## Training a forward model

A neural network learns the physics of the SpaceWorld using a pre-generated . The neural network's inputs are preprocessed in such a way that:
- Distances in the state vector are between 0 and 1
- Velocities in the state vector are between -1 and 1 

Actions are unchanged as they were already preprocessed in the dataset.


The network returns increments in position and velocity gained from leaving the initial state. The output is post-processed in such a way that:
- Position deltas and velocities are converted to their original pixel/pixel per timestep representation
- The next state is reconstituted by adding the increments to the initial state and by concatenating the constant part of the state vector. The post-processed output of the network has thus the exact same form than the state inputs found in the dataset.

Training generates three files:

- The .pt file containing the weights of the model
- A text file containing the initialisation parameters of the model 
- A _opti.pt file containing the weights of the optimizer  

Other files containing training data are also generated in the temp directory

- Checkpoints every 40 epochs in the temp/cps/ directory 
- rewardLoss and stateLoss which contains the reward and state loss at each iteration 
- tep_mse+suffix, tev_mse+suffix, ter_mse+suffix containing the position, velocity and reward MSE loss on the test set at each epoch

### Command

ssfmtr

### Parameters

- string **mp** : the path to the directory contraining your 8 dataset tensors. These tensors must keep their original names. Default: root directory.
- string **mdl** : the path and prefix to the file that will contain the trained model. Default: ../temp/model
- string **tag** : a suffix for some files generated during training. Default: root directory. 
- float **lr** : learning rate. Default: 0.001
- int **e** : Number of epochs. Default: 100
- int **bs** : batch size. Default: 32
- float **beta** : state loss multiplicative coefficient. Total loss = **beta** * stateloss + reward loss. Default: 1. 
- float **lp1** : The reward loss is penalized when it has a high error on rare events to speed training. This is a coefficient ranging from 0 to infinity. The higher the value, the more the model is penalized for having a high error when predicting a reward corresponding to the agent reaching the waypoint and turning the right signal on. Default: 0
- float **lp2** : This is a coefficient ranging from 0 to infinity. The higher the value, the more the model is penalized for having a high error when predicting a reward corresponding to the agent reaching the waypoint and turning the wrong signal on. Default: 0

### Example

You can train **myForward** on the datasets generated in **myMapPool** for **100** epochs with batches of size **128** and a learning rate of **0.0001**. You can speed up training by modifying the loss function by making the state loss **10** time more important or by penalizing the reward loss for its mistakes when predicting rare events by setting **lp1** and **lp2** to **0.1**

```
./project -cmd=ssmbfm -mdl=../temp/myForward -mp=../Starship/Maps/myMapPool -e=100 -bs=128 -lr=0.0001 -beta=10 -lp1=0.1 -lp2=0.1
```

## Computing the position error of a model

Print the position error a model makes on a test set. Also creates a file containing the error for each sample. You can use this file to, for example, plot a histogram. 

### Command

ssfmte

### Parameters

- string **mp** : the path to the directory contraining your 8 dataset tensors. These tensors must keep their original names. Default: root directory.
- string **mdl** : the path and prefix to the file that will contain the trained model. Default: ../temp/model
- string **f** : the path to the file that will be created. Default: root directory. 

## Testing action overfitting 

Computes the average position error your model makes with datasets with constant continuous actions values. This constant value varies from 0 to the maximum thrust. A file is then generated from which you can plot how the error varies when the constant changes.  

### Command 

ssaof

### Parameters 

- string **mp** : the path to the directory containing your train and test map pools. Default: root directory.
- int **nmaps** : the number of maps in the map pool. A value higher than the actual number of maps crashes the program. Default: 1
- int **n** : the number of trajectories. Overall, the dataset contrains **T** x **n** transitions. Default : 10000 
- int **T** : the number of transitions in a sample. **T** must divide 80 or the command won't execute itself. Default: 1
- float **wp** : Reaching a waypoint is a rare event. For the model to see this happen more often,  **wp** x **n** trajectories contain at least one transition towards a waypoint. **wp** should range from 0 to 1. Default: 0.1
- float **trp** : The training set's share of the dataset (ranging from 0 to 1). The testing set's share of the dataset is 1-**trp**. Default: 0.9
- float **sd** : The standard deviation of the gaussian distribution if **dist** is set to 2 or 3. Default: 0.25
- bool **woda** : you can disable the signal by setting this parameter to true. If so, the agent will receive a positive reward as long as he is on the waypoint. Default: false
- int **i** : the number of points you want in your graph. Default: 1
- string **mdl** : the path and prefix to the file that will contain the trained model. Default: ../temp/model
- string **f** : the path to the file that will be created. Default: root directory. 

You can use the generated file to make this kind of graph: 

![ssaof](https://github.com/jackdawe/joliRL/blob/master/img/ssaof.png)

As you can see, the error increases as the action value changes, which is a sign that your model is overfitting action wise. 

## Generating a seed for Gradient Based Planner (GBP)

You can provide a seed to GBP if you do not want it to randomly initialise the actions. This command will generate a random set of actions and save them in a .pt file. 

### Command

sssgen

### Parameters

- int **T** : the number of timesteps to unroll. Default: 1
- int **K** : the number of rollouts. Default: 1
- string **seed** : the path and filename of the tensor file. You should not put the ".pt" extension as it will be done automatically. Default: root directory.  

### Example

To generate an action seed to use GBP with **80** timesteps and **100** rollouts that will be saved as **mySeed**.pt in the temp directory: 

``` 
./project -cmd=sssgen -T=80 -K=100 -seed=../temp/mySeed 
```
## Launch a trial of Gradient Based Planner (GBP)

The agent appears on a map of your choice and uses GBP to plan actions for a whole episode. At the end of optimization, you can see the result on the GUI. 

### Command

ssmbplay

### Parameters

- string **mdl** : the path and prefix to the file that will contain the trained model. Default: ../temp/model
- string **map** : the path and name of the map on which you would like to run GBP.
- string **seed** : the path and filename to an action seed file. Default: "".  
- int **px** : the initial x coordinate of the ship. If left to default the ship spawns randomly. Default: -1
- int **py** : the initial y coordinate of the ship. If left to default the ship spawns randomly. Default: -1
- int **K** : the number of rollouts. Unnecessary if a seed is provided. Default: 1
- int **T** : the number of timesteps to unroll. Unnecessary if a seed is provided. Default: 1
- int **gs** : the number of gradient/optimization steps. Default: 1
- float **lr** : the learning rate. Default: 0.001
- bool **woda** : you can disable the signal by setting this parameter to true. If so, the agent will receive a positive reward as long as he is on the waypoint. Default: false

### Example

By executing the following command, an agent randomly appears on **map1** with **20** random sets of **80** actions. GBP uses the trained model **myForward.pt** for inference and performs **20** gradient steps with a learning rate of **0.01** and returns the set of optimized actions that leads to the highest reward. The GUI is then used to visualize the result.  

```
./project -cmd=ssmbplay -mdl=../temp/myForward -map=../Starship/Maps/myMapPool/test/map1 -K=20 -T=80 -lr=0.001 -gs=20
```
Now if instead you want to use your seed **mySeed** and have your ship start at (**100**,**200**) coordinates:

```
./project -cmd=ssmbplay -mdl=../temp/myForward -map=../Starship/Maps/myMapPool/test/map1 -seed=../temp/mySeed -px=100 -py=200 -lr=0.001 -gs=20
```
## Testing GBP over n trials 

The agent randonly spawns on a random map chosen from a map pool. GBP is used to predict the best sequence of actions. The obtained reward as well as the position error the model made are each written in a file. This process is repeated n times. You can then plot the results to assert the performance of GBP using your forward model.

### Command

ssmbtest

### Parameters

- string **mdl** : the path and prefix to the file that will contain the trained model. Default: ../temp/model
- string **mp** : the path to the directory containing the maps on which you wish to test gbp. Default: root directory. 
- string **f** : the path and prefix to the files that will be generated. Default: root directory. 
- int **K** : the number of rollouts. Unnecessary if a seed is provided. Default: 1
- int **T** : the number of timesteps to unroll. Unnecessary if a seed is provided. Default: 1
- int **gs** : the number of gradient/optimization steps. Default: 1
- float **lr** : the learning rate. Default: 0.001 
- bool **woda** : you can disable the signal by setting this parameter to true. If so, the agent will receive a positive reward as long as he is on the waypoint. Default: false

The files can be used to make these histograms: 

![Reward Plot](https://github.com/jackdawe/joliRL/blob/master/img/ssmbtest1.png)

![Error Plot](https://github.com/jackdawe/joliRL/blob/master/img/ssmbtest2.png)
