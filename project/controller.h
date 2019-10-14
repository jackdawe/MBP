#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "state.h"
#include "actionspace.h"

class Controller
{
public:
    Controller();
    virtual float transition(); //Changing the state after performing an action and returns the reward for that state action pair
    virtual bool isTerminal(State s); //returns true if the state is terminal
    virtual void generateStates(); //Converts the controller's representation of a state into a state vector readable by any agent. CurrentState and PreviousState are then updated
    virtual void updateStates();
    virtual int stateId(State s);
    virtual void reset();
    virtual vector<int> accessibleStates(State s);

    virtual int spaceStateSize();
    int actionSpaceSize();
    int saPairSpaceSize();

    ActionSpace getActions() const;
    State getPreviousState() const;
    vector<float> getTakenAction() const;
    float getTakenReward() const;
    State getCurrentState() const;
    vector<float> getRewardHistory() const;
    void addToRewardHistory(float r);
    void updateTakenAction(int actionIndex, float value);
    void setActions(const ActionSpace &value);
    void setTakenAction(const vector<float> &value);
    void setTakenReward(float value);

protected:
    ActionSpace actions;
    State previousState;
    vector<float> takenAction;
    float takenReward;
    State currentState;
    vector<float> rewardHistory;
};

#endif // CONTROLLER_H
