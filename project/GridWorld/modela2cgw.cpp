#include "modela2cgw.h"

ModelA2CGW::ModelA2CGW()
{    
}

ModelA2CGW::ModelA2CGW(int nInputs, int nHidden1, int nHidden2, int nHidden3):
nInputs(nInputs),nHidden1(nHidden1),nHidden2(nHidden2),nHidden3(nHidden3)
{
    fc1 = register_module("fc1",torch::nn::Linear(nInputs,nHidden1));
    fc1 = register_module("fc2",torch::nn::Linear(nHidden1,nHidden2));
    fc1 = register_module("fc3",torch::nn::Linear(nHidden2,nHidden3));
    actor = register_module("actor",torch::nn::Linear(nHidden3,4));
    critic = register_module("critic",torch::nn::Linear(nHidden3,1));
}

torch::Tensor ModelA2CGW::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return torch::relu(fc3->forward(x));
}

torch::Tensor ModelA2CGW::actorOutput(torch::Tensor x)
{
    return torch::softmax(actor->forward(x),1);
}

torch::Tensor ModelA2CGW::criticOutput(torch::Tensor x)
{
    return critic->forward(x);
}

