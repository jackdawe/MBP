# joliRL

This project aims to reproduce Mikael Henaff, Will Whitney and Yann LeCun's work on model-based planning.

In their research, they tested their algorithms on two tasks:

1. **GridWorld**, a state of the art toy example where the agent has to find its way in a grid by chosing whether to go up, down, left or right without colliding with an obstacle.

2. **Starship**, a more complex environment with an infinite state and action space. Many state of the art RL algorithms such as Q-Learning or TD-Lambda cannot be applied without using tricks such as discretizing the state space. The agent controls the thrusters of a small ship and has to fight the gravitational pull of the planets to reach the closest waypoint. Once the agent reaches the waypoint he has to fire the correct signal to indicate that he has reached this precise waypoint to collect a reward.  

For this work, I reproduced the environment as described in the paper. 

You can find all the details [here](https://arxiv.org/abs/1705.07177).

# General Information 

My code provides a certain amount of commands to easily use the main functionalities of this project. Every commands are documented down below.  

1. joliRL/project/ is the root directory.
2. To use one of my commands, first go to the root directory. Then follow the general architecture:
./project -cmd=COMMAND_NAME -myflag1= -myflag2= (...) -myflag100= 
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

# GRIDWORLD TASK

## Map Generation

Map files contain a simple matrix:
- 0 for an empty space
- 1 for an obstacle
- 2 for the goal
You can generate a GridWorld map of the size of your choice with this command. 


### Command name

gwmgen

### Parameters

- int **size**  : the size of the map. I advise you to stick to 8 and 16 as other sizes weren't tested. Default: 8 
- int **maxObst** : the maximum number of obstacles. The map will be generated with a random number of obstacles between 0 and maxObst. Please note that walls are automatically generated and are not taken into account in the maxObst count. Default: 1 
- string **map** : the path to the file that will be created. Default: root directory. 

### Example

To generate a map named myMap of size 16 with a maximum of 50 obstacles in the GridWorld/Maps/ directory, execute the following command:

./project -cmd=gwmgen -size=16 -maxObst=50 -map=../GridWorld/Maps/myMap 

#Map pool generation

For my algorithms to generalize well, I needed to train my agent on several maps. You can easily generate a pool of training and test maps using this command. 

### Command name

gwmpgen

### Parameters

- int **size** : the size of the map. I advise you to stick to 8 and 16 as other sizes weren't tested. Default: 8
- int **maxObst** : the maximum number of obstacles. The map will be generated with a random number of obstacles between 0 and maxObst. Please note that walls are automatically generated and are not taken into account in the maxObst count. Default: 1
- string **mp** : the path to the directory that will contain the test and train map pools. A new directory is created if it does not already exists. Default: root directory.
- int **nmaps**: the number of maps to be generated (both train and test map pools will contain nmaps maps). Default: 1

### Example 

To create a directory named MyMapPool in the GridWorld/Maps/ directory containing two sub directories (test and train) each containing 1000 maps of size 8 with a maximum of 10 obstacles:

./project -cmd=gwmpgen -size=8 -maxObst=10 -mp=../GridWorld/Maps/MyMapPool/ -nmaps=1000

## Showing the map using the implemented GUI

You can use the GUI to display the map on your screen.

### Command

**gwmshow**

### Parameters

- string **map** : the path to the file containing the map to be displayed. Default: root directory.

### Example

You can show the map named MyMap in the GridWorld/Maps/MyMapPool/test/map0 using the following command line:

./project -cmd=gwmshow -map=../GridWorld/Maps/MyMapPool/test/map0

Here is the type of window that you should get:

![Map example](https://github.com/jackdawe/joliRL/blob/master/img/Screenshot%20from%202020-01-16%2015-03-14.png "This is how the map should look like with Qt!")

## Generating the data set

An agent appears at a random location on a map randomly chosen from a map pool and wanders randomly until some conditions are met. Actions, State t, State t+1 and rewards are recorded and are each stored in a tensor.

Every sample contains T transitions. This is useful when you want your agent to learn only from the initial state, using its own predictions as inputs afterwards. You should set T to 1 if you do not want to use this feature as batches are made by picking random samples of T timesteps from the dataset. 

A train and test set are generated using the train and test maps respectively.

Executing this command created 8 files in the map pool directory:
- actionInputsTr.pt containing a { n x trp | T | 4 } tensor
- stateInputsTr.pt containing a { n x trp | T | size x size + 4 } tensor
- stateLabelsTr.pt containing a { n x trp | T | size x size + 4 } tensor
- rewardLabelsTr.pt containing a { n x trp | T } tensor
- actionInputsTe.pt containing a { n x (1-trp) | T | 4 } tensor
- stateInputsTe.pt containing a { n x (1-trp) | T | size x size + 4 } tensor
- stateLabelsTe.pt containing a { n x (1-trp) | T | size x size + 4 } tensor
- rewardLabelsTe.pt containing a { n x (1-trp) | T } tensor

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

The following command uses the 1000 first maps of the ../GridWorld/Maps/MyMapPool/ map pool directory to generate 8 tensor files forming both the test and training set. The tensor' first two dimension are trp*n x T for the training set and (1-trp)*n x T for the test set. 10% of the dataset contains samples where the agent appears next to the goal and makes a transition towards it. 

./project -cmd=gwdsgen -mp=../GridWorld/Maps/MyMapPool/ -nmaps=1000 -n=10000 -T=10 -wp=0.1 -trp=0.9 

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

To generate a map named myMap with 2 planets of radius ranging from 60 to 80, 4 waypoints of radius 15 in the Starship/Maps/ directory, execute the following command:

./project -cmd=ssmgen -nplan=2 -pmin=60 -pmax=80 -nwp=4 -rwp=15 -map=../Starship/Maps/myMap 

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

To create a directory named MyMapPool in the GridWorld/Maps/ directory containing two sub directories (test and train) each containing 100 maps using the default values for the planet and waypoint parameters:

./project -cmd=ssmpgen -mp=../Starship/Maps/myMapPool/ -nmaps=100 

## Showing the map using the implemented GUI

You can use the GUI to display the map on your screen.

### Command

**ssmshow**

### Parameters

- string **map** : the path to the file containing the map to be displayed. Default: root directory.

### Example

You can show the map named MyMap in the Starship/Maps/MyMapPool/test/map0 using the following command line:

./project -cmd=gwmshow -map=../Starship/Maps/MyMapPool/test/map0

Here is the type of window that you should get:

![Map example](https://github.com/jackdawe/joliRL/blob/master/img/Screenshot%20from%202020-01-16%2016-37-06.png "This is how the map should look like with Qt!")

The blue circle represents the planet. The other colored circles represent the waypoints.

## Generating the data set

An agent appears at a random location on a map randomly chosen from a map pool and randomly fires his thrusters and lights his signal for 80 timesteps. Actions, State t, State t+1 and rewards are recorded and are each stored in a tensor.

Every sample contains T transitions. This is useful when you want your agent to learn only from the initial state, using its own predictions as inputs afterwards. 

A train and test set are generated using the train and test maps respectively.

Action tensors are generated with discrete actions being encoded as one-hot vectors and thrust vector values are put between 0 and 1 as the planner will only take actions between these bounds. You can chose from 4 distributions for your actions:

#### Uniform over thrust vector coordinates (dist = 0)

[dist1](https://github.com/jackdawe/joliRL/blob/master/img/dist1.png)

#### Uniform over thrust norm and angle

[dist2](https://github.com/jackdawe/joliRL/blob/master/img/dist2.png)

#### Per trajectory isotopic gaussian over thrust vector coordinates

#### Per trajectory isotopic gaussian over thrust norm and angle

Executing this command created 8 files in the map pool directory:
- actionInputsTr.pt containing a { n x trp | T | a } tensor
- stateInputsTr.pt containing a { n x trp | T | s } tensor
- stateLabelsTr.pt containing a { n x trp | T | 4 } tensor
- rewardLabelsTr.pt containing a { n x trp | T } tensor
- actionInputsTe.pt containing a { n x (1-trp) | T | a } tensor
- stateInputsTe.pt containing a { n x (1-trp) | T | s } tensor
- stateLabelsTe.pt containing a { n x (1-trp) | T | 4 } tensor
- rewardLabelsTe.pt containing a { n x (1-trp) | T } tensor

s is the size of the state vector which is equal to 4 + 3 x (nPlanets + nWaypoints).

a is the size of the action vector which is equel to nWaypoints + 3. 

### Command

ssdsgen

### Parameters

- string **mp** : the path to the directory containing your train and test map pools. Default: root directory.
- int **nmaps** : the number of maps in the map pool. A value higher than the actual number of maps crashes the program. Default: 1
- int **n** : the size of the dataset. Overall, the dataset thus contrains T*n transitions. Default : 10000 
- int **T** : the number of transitions in a sample. If the agent reached a terminal state before the end of the T transitions, this same state is added to the State t and State t+1 tensors and 0 to the reward tensor. Default: 1
- float **wp** : Reaching a waypoint is a rare event. For the model to see this happen more often, the agent is forced spwan on a waypoint during wp*n episodes. Thus, wp should range from 0 to 1. This will only result in 2 to 3 transitions on the waypoint which is not much for an episode that lasts for 80 timesteps. Default: 0.1
- float **trp** : The training set's share of the dataset (ranging from 0 to 1). The testing set's share of the dataset is 1-trp. Default: 0.9

### Example

The following command uses the 1000 first maps of the ../GridWorld/Maps/MyMapPool/ map pool directory to generate 8 tensor files forming both the test and training set. The tensor' first two dimensions are trp*n x T for the training set and (1-trp)*n x T for the test set. 10% of the dataset contains samples where the agent appears next to the goal and makes a transition towards it. 

./project -cmd=ssdsgen -mp=../Starship/Maps/MyMapPool/ -nmaps=1000 -n=10000 -T=10 -wp=0.1 -trp=0.9 


## Training a forward model

A neural network learns the physics of the SpaceWorld using a pre-generated dataset using the ssdsgen command. The neural network's inputs are preprocessed in such a way that:
- Distances in the state vector are between 0 and 1
- Velocities in the state vector are between -1 and 1 
- Actions are unchanged as they were already preprocessed in the dataset
The network returns increment in position and velocity gained from leaving the initial state. The output is post-processed in such a way that:
- Position deltas and velocities are converted to their original pixel/pixel per timestep representation
- The next state is reconstituted by adding the increments to the initial state and by concatenating the constant part of the state vector. The post-processed output of the network has thus the exact same form than the state inputs found in the dataset.

Training generates two files:

- The .pt file containing the weights of the model
- A text file containing the initialisation parameters of the model 

### Command

ssmbfm

### Parameters

- string **mp** : the path to the directory contraining your 8 dataset tensors. These tensors must keep their original names. Default: root directory.
- string **mdl** : the path to the file that will contain the trained model.  
- int **depth??** :
- int **size??** : 
- float **lr** : learning rate. Default: 0.001
- int **n** : Number of iterations. Default: 10000
- int **bs** : batch size. Default: 32
- float **beta** : state loss multiplicative coefficient. Total loss = beta * stateloss + reward loss. Default: 1. 
- bool **asp** : If set to false, the neural network will only be provided with the initial state for each sample. The next states are then calculated using the model's previous predictions. Default: true.

# Full example on how to make the model based planner work

In this section I will give you a "to reproduce" example whith all the steps that you should take to make the model based planner perform well.
In this example, we take the standard case with one planet and three waypoints. 

## Step 1: Generate a map pool 

You first need to create maps on which your starship can fly. These maps will be used to generate your datasets. The more maps you have, the better your forward model will be able to generalise. 1000 maps is usually enough, but you can have more without any additionnal computationnal cost. 

./project -cmd=ssdsgen -mp=../Starship/Maps/MapPool1/ -nmaps=1000

By executing this command, the MapPool1 directory is created in the ../Starship/Maps/ directory. MapPool1 contains the test and train subdirectories whith 1000 maps each. The default values are used for the map parameters (waypoint and planet radii, etc...).

## Step 2: Generate your dataset using your map pool

We know want to use the maps we just created to make a dataset. You can use the following command to do so: 

./project -cmd=ssdsgen -mp=../Starship/Maps/MapPool1/ -nmaps=1000 -n=250000 -T=40 -wp=0.5 -trp=0.99

With the first two flags you indicate that you want to use the map pool you previously generated.
The n flag will determine the size of your dataset. A rather large dataset is required to prevent overfitting.

The T flag determines how many timesteps there are in one data sample. We set T to 40 so that the neural network can learn to take into account its errors while predicting the next state and reward. Training is way more challenging that way, but prediction errors not build up way less during inference, even if you want to predict 80 timesteps ahead.
Since an episode lasts for 80 timesteps, your dataset will contain 125000 episodes.
If you ask the ship to wander randomly on a map, the probability of encountering a waypoint is very low. As a result, the forward model will have alot of trouble to predict a positive reward. By setting wp to 0.5, I ask my dataset to only keep samples that contain at least one encounter with a waypoint for 50% of the dataset.
Finally, 2500 samples (100 000 transitions) is enough for the test set, thus I set trp to 0.99.

Please note that this command does not use the GPU. The higher n, T and wp the longer it will take to execute. Generating such a dataset can take up to 2 hours. Lowering wp greatly reduces computational time.

## Step 3: Learn a forward model

My neural network will use the dataset generated in step 2 to learn the transition and reward function of the space world.

./project -cmd=ssmbfm -mp=../Starship/Maps/MapPool1/ -mdl=../myForward -tag?? -lr=0.0001 -bs=128 -n=100 -wn -sd=0.25

You should give the directory containing your 8 ".pt" dataset files to the mp flag.
By setting mdl to "../myForward", two files will be created and updated in the root's parent directory: myForward.pt containing the weights of the model and myForward_Params containing the parameters that you set to create the model. 
lr and bs are hyperparameters that you can change, but training goes well with the ones I chose.
wn enables white noise to be added to the discrete action one-hot encoding. sd represents the standard deviation of the white noise. The white noise is supposed to smooth the reward function in the space between the one-hot encoded vectors in such a way that optimisation using gradient descent works better. If sd is too low, there will be little effect. If it is too high, there will be a non-negligeable probability that the original action value becomes lower than another one, thus creating an inconsistency in the dataset.

Every n forward passes, training pauses to do the following operations:

- Replacing the myForward.pt file with a file containing the updated weights
- Printing the performance of the network on the test set
- Saving some training data such as state and reward loss for you to plot

Training then resumes for n other forward passes. 

In this configuration, you should see less than 20 pixel position error after 30 minutes of training. If left for a few hours, you should be able to have less than 10 pixel of position error on all the test set.

## Step 4: Use the forward model to plan your actions 

[TUTORIAL] Making you own world class

A world class describes the rules of your world: the environment, the state encoding, the possible actions, how an agent is rewarded for performing a specific transition and so on.

Your world class should be a children of my World class and implement the following methods:

- float transition()
- bool isTerminal(State s)
- void generateVectorStates()
- void reset() 

[TUTORIAL] How to use model based planning with your own world?

My code includes a ModelBased class that you can use to plan with your own worlds. To initialise a ModelBased object, you will need to provide your neural network class.    
Model based planning is divided into 2 main steps: training your neural network and action planning using your learnt forward model.

LEARNING THE FORWARD MODEL

The ModelBased class has a method that you can use to train your forward model using a dataset that you should provide. Here is the prototype:

learnForwardModel(torch::optim::Adam *optimizer, torch::Tensor actionInputs, torch::Tensor stateInputs, torch::Tensor stateLabels, torch::Tensor rewardLabels, int epochs, int batchSize=32, float beta=1, bool allStatesProvided = true);

BUILDING THE DATASET

You are free on the method to build your dataset. For compatibility purpose, your dataset tensors should respect these few conditions:

1. All 4 tensors must have at least 2 dimensions. Dimension 0 contains samples that each contain T transitions. Thus dimension 1 should be of size T.
2. stateInputs 

To use the model based planner, you will have to build your own forward model class. It will have to fulfill some requirements for it to be fully compatible with the ModelBased class.

To help you, I have provided a mother class (forward.h forward.cpp) from which your class should be derived. You'll have to override the forward and computeLoss methods:

### forward

Prototype: void forward(torch::Tensor stateBatch, torch::Tensor actionBatch, bool restore=false)

This method takes as input a batch of states and actions and should update the predictedState and predictedReward attributes inherited from the ForwardImpl class. Your are free to use this method to encode / decode your data to meet the requirements of the ModelBased class. You might need, for example, to normalize your data or encode your states as images if your model uses convolutions.

stateBatch and actionBatch's dimension 0 and 1 have been merged in such a way that new dimension 0's size is batchSize*T. predictedState and predictedReward should have this same dimension 0 size.

### computeLoss

Prototype: virtual void computeLoss(torch::Tensor stateLabels, torch::Tensor rewardLabels)

This method should use the input labels as well as the predictedState and predictedReward attributes to update the stateLoss and rewardLoss attributes inherited from the ForwardImpl class. These losses are then used by ModelBased to perform a backward pass through your forward model and update its weights. stateLoss and rewardLoss should be scalar tensors.

stateLabels and rewardLabels are batches from your provided dataset also have their two first dimension merged.

The ForwardImpl method also provides a public usedDevice attribute that contains the nature of the device on which the training will be done on (GPU or CPU). usedDevice will be updated upon initialisation of an object of your forward model class.    


-Having your own nn class: must have forward method, must have compute loss method, must have predictedState and predictedReward public attribute, 
-Adding to the template
-Initialise an optimizer (/!\ only Adam atm)
-Restrictions on the dataset
-can use method to save loss data
