#include "randomagent.h"

template <class C>
RandomAgent<C>::RandomAgent()
{
}

template <class C>
RandomAgent<C>::RandomAgent(C controller): Agent<C>(controller,1)
{
}

template class RandomAgent<ControllerSS>;
