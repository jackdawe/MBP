#include "state.h"

State::State()
{
    generateStateVector();
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
    stateVector = stateVector;
}

vector<int> State::accessibleStates()
{
}
