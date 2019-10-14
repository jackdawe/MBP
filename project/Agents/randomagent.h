#ifndef RANDOMAGENT_H
#define RANDOMAGENT_H
#include "agent.h"
#include "Starship/controllerss.h"

template <class C>
class RandomAgent: public Agent<C>
{
public:
    RandomAgent();
    RandomAgent(C controller);
};

#endif // RANDOMAGENT_H
