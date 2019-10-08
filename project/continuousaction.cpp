#include "continuousaction.h"

ContinuousAction::ContinuousAction()
{

}

ContinuousAction::ContinuousAction(double lowerBound, double upperBound):lowerBound(lowerBound), upperBound(upperBound)
{

}

double ContinuousAction::pick()
{
    default_random_engine generator(random_device{}());
    uniform_real_distribution<double> dist(lowerBound,upperBound);
    return dist(generator);
}

double ContinuousAction::getLowerBound() const
{
    return lowerBound;
}

double ContinuousAction::getUpperBound() const
{
    return upperBound;
}
