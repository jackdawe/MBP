#ifndef STATE_H
#define STATE_H
#include "action.h"
#include <opencv2/opencv.hpp>
class State
{
public:
    State();
    State(vector<float> stateVector);
    void add(float value);
    void update(int index, float value);
    vector<float> currentStateVector();

    vector<float > getStateVector() const;
    void setStateVector(const vector<float > &value);

private:
    vector<float> stateVector;
};

#endif // STATE_H
