#include "state.h"

State::State()
{
}

void State::add(double *value)
{
    stateVector.push_back(value);
}
