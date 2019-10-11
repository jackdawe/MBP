#ifndef CONTINUOUSACTION_H
#define CONTINUOUSACTION_H
#include "action.h"

class ContinuousAction: public Action
{
public:
    ContinuousAction();
    ContinuousAction(float lowerBound, float upperBound);
    float pick();
    float getLowerBound() const;

    float getUpperBound() const;

private:
    float lowerBound;
    float upperBound;
};

#endif // CONTINUOUSACTION_H
