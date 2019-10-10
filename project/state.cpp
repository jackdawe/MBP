#include "state.h"

State::State()
{
}

void State::add(double value)
{
    stateVector.push_back(value);
}

void State::update(int index, double value)
{
    stateVector[index] = value;
}

vector<double> State::getStateVector() const
{
    return stateVector;
}

void State::setStateVector(const vector<double> &value)
{
    stateVector = value;
}
