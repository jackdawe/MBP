#include "state.h"

State::State()
{
    generateStateVector();
}

void State::transition(Action a)
{
}

double State::reward(Action a)
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
