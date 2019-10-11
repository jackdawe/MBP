#include "continuousaction.h"

ContinuousAction::ContinuousAction()
{

}

ContinuousAction::ContinuousAction(float lowerBound, float upperBound):lowerBound(lowerBound), upperBound(upperBound)
{

}

float ContinuousAction::pick()
{
    default_random_engine generator(random_device{}());
    uniform_real_distribution<float> dist(lowerBound,upperBound);
    return dist(generator);
}

float ContinuousAction::getLowerBound() const
{
    return lowerBound;
}

float ContinuousAction::getUpperBound() const
{
    return upperBound;
}
