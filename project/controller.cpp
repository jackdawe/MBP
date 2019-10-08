#include "controller.h"

Controller::Controller()
{

}

void State::transition(double a)
{
}

double State::reward(double a)
{
    return 0;
}

bool State::isTerminal()
{
    return false;
}

void State::generateStateVector()
{
}

vector<int> State::accessibleStates(State s)
{
}
