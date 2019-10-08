#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "state.h"
#include "actionspace.h"

class Controller
{
public:
    Controller();
private:
    ActionSpace actions;
    State previousState;
    vector<double> takenAction;
    double takenReward;
    State currentState;

    vector<double> rewardHistory;
};

#endif // CONTROLLER_H
