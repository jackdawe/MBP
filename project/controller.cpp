#include "controller.h"

Controller::Controller():
    takenAction(vector<double>(actions.size(),0))
{

}

double Controller::transition(double a)
{
}

bool Controller::isTerminal()
{
    return false;
}

void Controller::generateStateVector()
{
}

vector<int> Controller::accessibleStates(State s)
{
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
