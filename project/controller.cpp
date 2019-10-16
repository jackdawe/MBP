#include "controller.h"

Controller::Controller()
{
}

float Controller::transition()
{
    return 0;
}

bool Controller::isTerminal(State s)
{
    return false;
}

void Controller::generateStates()
{
}

int Controller::stateId(State s)
{
    return -1;
}

void Controller::reset()
{

}

vector<int> Controller::accessibleStates(State s)
{
    return {-1};
}

int Controller::spaceStateSize()
{
    return -1;
}

int Controller::actionSpaceSize()
{
    return actions.cardinal();
}

int Controller::saPairSpaceSize()
{
    return spaceStateSize()*actionSpaceSize();
}

ActionSpace Controller::getActions() const
{
    return actions;
}

State Controller::getPreviousState() const
{
    return previousState;
}

vector<float> Controller::getTakenAction() const
{
    return takenAction;
}

float Controller::getTakenReward() const
{
    return takenReward;
}

State Controller::getCurrentState() const
{
    return currentState;
}

vector<float> Controller::getRewardHistory() const
{
    return rewardHistory;
}

void Controller::addToRewardHistory(float r)
{
    rewardHistory.push_back(r);
}

void Controller::updateTakenAction(int actionIndex, float value)
{
    takenAction[actionIndex] = value;
}

void Controller::setActions(const ActionSpace &value)
{
    actions = value;
}

void Controller::setTakenAction(const vector<float> &value)
{
    takenAction = value;
}

void Controller::setTakenReward(float value)
{
    takenReward = value;
}

vector<string> Controller::getParamLabels() const
{
    return paramLabels;
}

vector<float> Controller::getParamValues() const
{
    return paramValues;
}

string Controller::getPath() const
{
    return path;
}
