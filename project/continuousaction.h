#ifndef CONTINUOUSACTION_H
#define CONTINUOUSACTION_H
#include "action.h"

class ContinuousAction: public Action
{
public:
    ContinuousAction();
    ContinuousAction(double lowerBound, double upperBound);
private:
    double lowerBound;
    double upperBound;
};

#endif // CONTINUOUSACTION_H
