#include "controller.h"

Controller::Controller():
    takenAction(vector<double>(actions.size(),0))
{
}

double Controller::transition()
{
    return 0;
}

bool Controller::isTerminal()
{
    return false;
}

void Controller::generateStateVector()
{
}

int Controller::stateId(State s)
{
    return 0;
}

vector<int> Controller::accessibleStates(State s)
{
    return vector<int>();
}

int Controller::spaceStateSize()
{
    return 0;
}

ActionSpace Controller::getActions() const
{
    return actions;
}

State Controller::getPreviousState() const
{
    return previousState;
}

vector<double> Controller::getTakenAction() const
{
    return takenAction;
}

double Controller::getTakenReward() const
{
    return takenReward;
}

State Controller::getCurrentState() const
{
    return currentState;
}

vector<double> Controller::getRewardHistory() const
{
    return rewardHistory;
}

void Controller::updateTakenAction(int actionIndex, double value)
{
    takenAction[actionIndex] = value;
}

void Controller::setActions(const ActionSpace &value)
{
    actions = value;
}

void Controller::setTakenAction(const vector<double> &value)
{
    takenAction = value;
}

void Controller::setTakenReward(double value)
{
    takenReward = value;
}
