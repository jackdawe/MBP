#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "state.h"
#include "actionspace.h"

class Controller
{
public:
    Controller();
    virtual double transition(double a); //Changing the state after performing an action and returns the reward for that state action pair
    virtual bool isTerminal(); //returns true if the state is terminal
    virtual void generateStateVector(); //Converts any representation of a state to a state vector readable by the agentTrainer
    virtual vector<int> accessibleStates(State s);

    ActionSpace getActions() const;
    State getPreviousState() const;
    vector<double> getTakenAction() const;
    double getTakenReward() const;
    State getCurrentState() const;
    vector<double> getRewardHistory() const;
    void updateTakenAction(int actionIndex, double value);

protected:
    ActionSpace actions;
    State previousState;
    vector<double> takenAction;
    double takenReward;
    State currentState;
    vector<double> rewardHistory;
};

#endif // CONTROLLER_H
