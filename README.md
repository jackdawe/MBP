# joliRL

This project aims to reproduce Mikael Henaff, Will Whitney and Yann LeCun's work on model-based reinforcement learning.

In their research, they tested their algorithms on two tasks:

1. **GridWorld**, a state of the art toy example where the agent has to find its way in a grid by chosing whether to go up, down, left or right without colliding with an obstacle.

2. **Starship**, a more complex environment with an infinite state and action space. Many state of the art RL algorithms such as Q-Learning or TD-Lambda cannot be applied without using tricks such as discretizing the state space. The agent controls the thrusters of a small ship and has to fight the gravitational pull of the planets to reach the closest waypoint. Once the agent reaches the waypointm he has to fire a signal indicating that he has reached this precise waypoint to collect a reward.  

For this work, I reproduced the environment as described in the paper. 
You can find all the details [here](https://arxiv.org/abs/1705.07177).

# GRIDWORLD TASK

## Map Generation

Map files contain a simple matrix: 0 is an empty space, 1 an obstacle and 2 the goal. To generate a map, you can use the **gwmgen** command with the following parameters:

### Command

gwmgen

### Parameters

- int **size**  : the size of the map. I advise you to stick to 8 and 16 as other sizes weren't tested. Default: 8 .
- int **maxObst** : the maximum number of obstacles. The map will be generated with a random number of obstacles between 0 and maxObst. Please note that walls are automatically generated and are not taken into account in the maxObst count. Default: 1 .
- string **map** : the path to the file that will be created. Default: root directory. 

### Example

To generate a map named myMap of size 16 with a maximum of 50 obstacles in the GridWorld/Maps/ directory, execute the following command:

./project -cmd=gwmgen -size=16 -maxObst=50 -map=../GridWorld/Maps/myMap 

#Map pool generation

For my algorithms to generalize well, I needed to train my agent on several maps. The **gwmpgen** command allows to easily generate a pool of training and test maps using this set of parameters:

### Command

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

## Generating the data set

An agent appears at a random location on a map randomly chosen from a map pool and wanders randomly until some conditions are met. Actions, State t, State t+1 and rewards are recorded and are each stored in a tensor .
Every sample contains T transitions. This is useful when you want your agent to learn only from the initial state, using its own predictions as inputs afterwards.
A train and test set are generated using the train and test maps respectively.
Executing this command created 8 files in the map pool directory: actionInputsTr.pt, stateInputsTr.pt, stateLabelsTr.pt, rewardLabelsTr.pt, actionInputsTe.pt, stateInputsTe.pt, stateLabelsTe.pt, rewardLabelsTe.pt . 

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

- string **mp** : the path to the directory contraining your 8 dataset tensors. These tensors must keep their default names. Default: root directory.
- int **sc1** : manages the number of feature maps of the convolutionnal layers. Default: 16
- float **lr** : learning rate. Default: 0.001
- int **n** : Number of iterations. Default: 10000
- int **bs** : batch size. Default: 32
- float **beta** : state loss multiplicative coefficient. Total loss = beta * stateloss + reward loss. Default: 1. 
- bool **asp** : If set to true, the neural network will only be provided with the initial state for each sample. The next states are then calculated using the model's previous predictions. Default: true.

# Gradient Based Planner

Command: ???
Parameters:
- K : The number of rollouts.
- T : The amount of time steps to be planned
- gs : The number of gradient steps to be performed
- lr : Learning rate

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
