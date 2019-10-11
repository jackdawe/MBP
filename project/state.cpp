#include "state.h"

State::State()
{
}

void State::add(float value)
{
    stateVector.push_back(value);
}

void State::update(int index, float value)
{
    stateVector[index] = value;
}

vector<float> State::getStateVector() const
{
    return stateVector;
}

void State::setStateVector(const vector<float> &value)
{
    stateVector = value;
}
