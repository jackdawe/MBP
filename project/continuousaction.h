#ifndef CONTINUOUSACTION_H
#define CONTINUOUSACTION_H
#include "action.h"

class ContinuousAction: public Action
{
public:
    ContinuousAction();
    ContinuousAction(double lowerBound, double upperBound);
    double pick();
    double getLowerBound() const;

    double getUpperBound() const;

private:
    double lowerBound;
    double upperBound;
};

#endif // CONTINUOUSACTION_H
