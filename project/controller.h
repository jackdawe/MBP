#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "state.h"
#include "actionspace.h"

class Controller
{
public:
    Controller();
    virtual void transition(double a); //Changing the state after performing a
    virtual double reward(double a); //Returns the reward for taking action a
    virtual bool isTerminal(); //returns true if the state is terminal
    virtual void generateStateVector(); //Converts any representation of a state to a state vector readable by the agentTrainer
    virtual vector<int> accessibleStates(State s);
private:
    ActionSpace actions;
    State previousState;
    vector<double> takenAction;
    double takenReward;
    State currentState;

    vector<double> rewardHistory;
};

#endif // CONTROLLER_H
